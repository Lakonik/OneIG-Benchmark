from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error, setup_distributed

from scripts.text.text_utils import preprocess_string, clean_and_remove_hallucinations, levenshtein_distance, calculate_char_match_ratio
from scripts.utils.inference import Qwen2_5VLBatchInferencer
import tempfile
import torch
import torch.distributed as dist
import socket
import json

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def main():
    args = parse_args()
    
    ddp_active, world_size, rank, local_rank, device = setup_distributed()

    cache_dir = os.path.join(tempfile.gettempdir(), f"oneigbench_tmp_{formatted_time}_rank{rank}")
    os.makedirs(cache_dir, exist_ok=True)


    # every rank loads (and if needed downloads) its own copy; HF hub downloads are
    # serialized by file locks, so no cross-rank barrier is required
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
    
    # Local accumulators for distributed reduction, kept per model so multi-model
    # evaluations do not pool every model's samples into one shared score
    local_model_stats = {}  # model_name -> (ed, cr, match_word, gt_word) sample lists
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
                if len(split_img_list) == 0:  # e.g. an all-black grid was filtered out
                    local_prompt_results.append((id, model_name, None))
                    continue
                ocr_results = influencer.infer_ocr(split_img_list, max_new_tokens)

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

        local_model_stats[model_name] = (
            edit_distances, completion_ratios, match_word_counts, gt_word_counts)

    # Gather all local aggregates
    if ddp_active:
        gathered_prompts = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_prompt_results, gathered_prompts, dst=0)
        gathered_stats = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_model_stats, gathered_stats, dst=0)
    else:
        gathered_prompts = [local_prompt_results]
        gathered_stats = [local_model_stats]

    if rank == 0:
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for part in gathered_prompts:
            for id, model_name, triple in part:
                score_of_prompt_csv.loc[id, model_name] = triple
        for model_name in args.model_names:
            edit_distances, completion_ratios, match_word_counts, gt_word_counts = [], [], [], []
            for part in gathered_stats:
                ed, cr, mwc, gtwc = part.get(model_name, ((), (), (), ()))
                edit_distances.extend(ed)
                completion_ratios.extend(cr)
                match_word_counts.extend(mwc)
                gt_word_counts.extend(gtwc)
            ED = sum(edit_distances) / len(edit_distances) if len(edit_distances) else 0.0
            CR = sum(completion_ratios) / len(completion_ratios) if len(completion_ratios) else 0.0
            WAC = (sum(match_word_counts) / sum(gt_word_counts)) if sum(gt_word_counts) else 0.0
            score_csv.loc[model_name, "ED"] = ED
            score_csv.loc[model_name, "CR"] = CR
            score_csv.loc[model_name, "WAC"] = WAC
            score_csv.loc[model_name, "text score"] = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

        save2csv(score_csv, text_score_csv)

        # Print parseable final results on rank 0
        result_dict = {}
        for model_name in args.model_names:
            row = score_csv.loc[model_name].to_dict()
            row = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            result_dict[model_name] = row
        print("FINAL_RESULT " + json.dumps({
            "script": "text",
            "mode": args.mode,
            "timestamp": formatted_time,
            "results": result_dict
        }))

    # save2csv(score_of_prompt_csv, text_prompt_score_csv)

    if ddp_active:
        dist.barrier()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()