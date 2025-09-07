import scripts.reasoning.restore_llama as r
r.apply()

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

import json
from scripts.utils.inference import LLM2CLIP
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

    # Per-rank cache to avoid shared-FS issues
    base_cache = os.path.join(tempfile.gettempdir(), f"oneig_cache_rank{rank}")
    os.makedirs(base_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(base_cache, "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(base_cache, "hf_hub_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base_cache, "transformers_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base_cache, "xdg_cache"))

    LLM2CLIP_Model = LLM2CLIP()
    
    if args.mode == "EN":
        answer_json_dir = "scripts/reasoning/gt_answer.json"
    else:
        answer_json_dir = "scripts/reasoning/gt_answer_zh.json"
    with open(answer_json_dir, 'r', encoding='utf-8') as f:
        answer_gt = json.load(f)
        
    reasoning_score_csv = f"results/reasoning_score_{args.mode}_{formatted_time}.csv"
    reasoning_prompt_score_csv = f"results/reasoning_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(reasoning_score_csv), exist_ok=True)
    
    score_csv = pd.DataFrame(index=args.model_names, columns=["reasoning"]) if rank == 0 else None
    
    # Model per rank with rank-0 warm-up (HF downloads)
    # Decide warm-up leader: shared cache uses global rank 0, otherwise per-node GPU0
    base_cache = os.path.join(tempfile.gettempdir(), f"oneig_cache_rank{rank}")
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
            LLM2CLIP_Model = LLM2CLIP()
        dist.barrier()
        if not is_leader:
            LLM2CLIP_Model = LLM2CLIP()
    else:
        LLM2CLIP_Model = LLM2CLIP()
    
    # Local results list: (img_id, model_name, avg_score)
    local_results = []

    for model_id, model_name in enumerate(args.model_names):
        
        if rank == 0:
            print(f"It is {model_name} time.")
        
        img_grid = (args.image_grid[model_id], args.image_grid[model_id]) 
        
        image_dir = args.image_dirname + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        img_list = sorted(img_list)
        
        if rank == 0:
            print(f"We fetch {len(img_list)} images.")
        
        # Shard across ranks
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images", disable=(rank != 0)):
            if idx % (world_size if world_size > 0 else 1) != rank:
                continue
            
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            
            img_id = img_path.split('/')[-1][:3]
            answer_text = answer_gt[img_id]

            score = LLM2CLIP_Model.text_img_similarity_score(split_img_list, answer_text)

            if len(score) != 0:
                score = [x for x in score if x is not None]
                local_results.append((img_id, model_name, sum(score)/len(score)))
            else:
                local_results.append((img_id, model_name, None))
    # Gather results
    if ddp_active:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
    else:
        gathered = [local_results]

    if rank == 0:
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for part in gathered:
            for img_id, model_name, val in part:
                score_of_prompt_csv.loc[img_id, model_name] = val
        mean_values = score_of_prompt_csv.mean()
        score_csv["reasoning"] = mean_values.values
        save2csv(score_csv, reasoning_score_csv)
    
    # score_of_prompt_csv = score_of_prompt_csv.sort_index()
    # save2csv(score_of_prompt_csv, reasoning_prompt_score_csv)

    if ddp_active:
        dist.barrier()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

    if rank == 0:
        # Print parseable final results on rank 0
        result_dict = {}
        for model_name in args.model_names:
            row = score_csv.loc[model_name].to_dict()
            row = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            result_dict[model_name] = row
        print("FINAL_RESULT " + json.dumps({
            "script": "reasoning",
            "mode": args.mode,
            "timestamp": formatted_time,
            "results": result_dict
        }))

if __name__ == "__main__":
    main()
            