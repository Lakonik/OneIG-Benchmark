from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

from scripts.text.text_utils import preprocess_string, clean_and_remove_hallucinations, levenshtein_distance, calculate_char_match_ratio
from scripts.utils.inference import Qwen2_5VLBatchInferencer
import tempfile
import torch
import torch.distributed as dist
import socket

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def main():
    args = parse_args()
    
    # Initialize distributed environment if available
    def _dist_init_if_needed():
        if dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                dist.init_process_group(backend=backend, init_method="env://")
            except Exception:
                pass

    _dist_init_if_needed()

    ddp_active = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if ddp_active else 1
    rank = dist.get_rank() if ddp_active else 0
    local_rank = dist.get_node_local_rank() if ddp_active else 0

    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
        except Exception:
            device_index = local_rank
        device = f"cuda:{device_index}"
    else:
        device = "cpu"

    cache_dir = os.path.join(tempfile.gettempdir(), f"oneigbench_tmp_{formatted_time}_rank{rank}")
    os.makedirs(cache_dir, exist_ok=True)

    # Per-rank cache roots to avoid shared-FS conflicts
    base_cache = os.path.join(tempfile.gettempdir(), f"oneig_cache_rank{rank}")
    os.makedirs(base_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(base_cache, "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(base_cache, "hf_hub_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base_cache, "transformers_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base_cache, "xdg_cache"))

    # Instantiate VLM with rank-0 warm-up to avoid concurrent downloads
    # Decide warm-up leader based on path sharing
    def _is_within_repo(path):
        try:
            repo_root = os.path.abspath(os.getcwd())
            ap = os.path.abspath(path)
            return ap.startswith(repo_root + os.sep)
        except Exception:
            return False
    shared_download = _is_within_repo(base_cache)
    if ddp_active:
        is_leader = (rank == 0) if shared_download else (local_rank == 0)
        if is_leader:
            influencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
        dist.barrier()
        if not is_leader:
            influencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
    else:
        influencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
    
    if args.mode == "EN":
        text_csv_path = "scripts/text/text_content.csv"
        MAX_EDIT_DISTANCE = 100
    else:
        text_csv_path = "scripts/text/text_content_zh.csv"
        MAX_EDIT_DISTANCE = 50
    text_df = pd.read_csv(text_csv_path, dtype=str)

    text_score_csv = f"results/text_score_{args.mode}_{formatted_time}.csv"
    text_prompt_score_csv = f"results/text_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(text_score_csv), exist_ok=True)
    
    score_csv = pd.DataFrame(index=args.model_names, columns=["ED", "CR", "WAC", "text score"]) if rank == 0 else None
    
    # Local accumulators for distributed reduction
    local_edit_distances = []
    local_completion_ratios = []
    local_match_word_counts = []
    local_gt_word_counts = []
    local_prompt_results = []  # (id, model_name, [ED_mean, CR_mean, WAC_mean])

    for model_id, model_name in enumerate(args.model_names):
        
        if rank == 0:
            print(f"It is {model_name} time.")
        
        img_grid = (args.image_grid[model_id], args.image_grid[model_id]) 
        
        edit_distances = []
        completion_ratios = []
        match_word_counts = []
        gt_word_counts = []
        
        for idx_all, (id, text_gt) in tqdm(enumerate(zip(text_df["id"], text_df["text_content"])), total=len(text_df), desc="Processing text", disable=(rank != 0)):
            if idx_all % (world_size if world_size > 0 else 1) != rank:
                continue
            word_count = len(text_gt.split())
            if (word_count > 60):
                max_new_tokens = 256
            else:
                max_new_tokens = 128
                
            text_gt_preprocessed = preprocess_string(text_gt)
            
            img_path = megfile.smart_glob(args.image_dirname + '/' + model_name + '/' +  id + '*')
            if len(img_path) != 1:
                local_prompt_results.append((id, model_name, None))
            else:
                split_img_list = split_2x2_grid(img_path[0], img_grid, cache_dir)    
                if  len(split_img_list) != 0:                 
                    ocr_results = influencer.infer_ocr(split_img_list, max_new_tokens)
                else:
                    local_prompt_results.append((id, model_name, None))
                
                text_ocr_list = clean_and_remove_hallucinations(ocr_results)
                
                ED_score = []
                CR_score = []
                WAC_score = []
                
                for text_ocr in text_ocr_list:
                    text_ocr_preprocessed = preprocess_string(text_ocr)
                    
                    edit_distance = levenshtein_distance(text_ocr_preprocessed, text_gt_preprocessed)
                    
                    completion_ratio = 1 if edit_distance == 0 else 0
                    
                    match_word_count, text_word_accuracy, gt_word_count = calculate_char_match_ratio(text_gt_preprocessed, text_ocr_preprocessed)
                    
                    edit_distances.append(edit_distance)
                    completion_ratios.append(completion_ratio)
                    match_word_counts.append(match_word_count)
                    gt_word_counts.append(gt_word_count)

                    ED_score.append(edit_distance)
                    CR_score.append(completion_ratio)
                    WAC_score.append(text_word_accuracy)

                local_prompt_results.append((id, model_name, [float(sum(ED_score)/len(ED_score)), float(sum(CR_score)/len(CR_score)), float(sum(WAC_score)/len(WAC_score))]))

        local_edit_distances.extend(edit_distances)
        local_completion_ratios.extend(completion_ratios)
        local_match_word_counts.extend(match_word_counts)
        local_gt_word_counts.extend(gt_word_counts)

    # Gather all local aggregates
    if ddp_active:
        gathered_prompts = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_prompt_results, gathered_prompts, dst=0)
        gathered_ed = [None for _ in range(world_size)] if rank == 0 else None
        gathered_cr = [None for _ in range(world_size)] if rank == 0 else None
        gathered_mwc = [None for _ in range(world_size)] if rank == 0 else None
        gathered_gtwc = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_edit_distances, gathered_ed, dst=0)
        dist.gather_object(local_completion_ratios, gathered_cr, dst=0)
        dist.gather_object(local_match_word_counts, gathered_mwc, dst=0)
        dist.gather_object(local_gt_word_counts, gathered_gtwc, dst=0)
    else:
        gathered_prompts = [local_prompt_results]
        gathered_ed = [local_edit_distances]
        gathered_cr = [local_completion_ratios]
        gathered_mwc = [local_match_word_counts]
        gathered_gtwc = [local_gt_word_counts]

    if rank == 0:
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for part in gathered_prompts:
            for id, model_name, triple in part:
                score_of_prompt_csv.loc[id, model_name] = triple
        edit_distances = [x for part in gathered_ed for x in part]
        completion_ratios = [x for part in gathered_cr for x in part]
        match_word_counts = [x for part in gathered_mwc for x in part]
        gt_word_counts = [x for part in gathered_gtwc for x in part]

        ED = sum(edit_distances) / len(edit_distances) if len(edit_distances) else 0.0
        CR = sum(completion_ratios) / len(completion_ratios) if len(completion_ratios) else 0.0
        WAC = (sum(match_word_counts) / sum(gt_word_counts)) if sum(gt_word_counts) else 0.0
        
        score_csv.loc[args.model_names, "ED"] = None  # initialize rows
        for model_name in args.model_names:
            score_csv.loc[model_name, "ED"] = ED
            score_csv.loc[model_name, "CR"] = CR
            score_csv.loc[model_name, "WAC"] = WAC
            score_csv.loc[model_name, "text score"] = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

        save2csv(score_csv, text_score_csv)

    # save2csv(score_of_prompt_csv, text_prompt_score_csv)

    if ddp_active:
        dist.barrier()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()