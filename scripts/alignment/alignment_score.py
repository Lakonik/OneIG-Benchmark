from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

import json
from copy import deepcopy
import tempfile
import torch
import torch.distributed as dist
from scripts.utils.inference import Qwen2_5VLBatchInferencer

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def alignment_score(img_path, questions, dependencies, img_grid, cache_dir, inferencer):
    score = {}
    
    if len(img_path) == 1:
        split_img_list = split_2x2_grid(img_path[0], img_grid, cache_dir)
        if len(split_img_list) == 0:
            return None    
    else:
        return None
    
    for id, question in questions.items():
        images_path = split_img_list
        batch_answer = inferencer.infer_semantic(images_path, question)
        score[id] = [float(ans == "Yes") for ans in batch_answer]
        
    filter_score = deepcopy(score)
    for img_idx in range(len(split_img_list)):
        for id, parent_ids in dependencies.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                try:
                    if score[parent_id][img_idx] == 0:
                        any_parent_answered_no = True
                        break
                    else:
                        continue
                except:
                    print("The score is not a number.")
            if any_parent_answered_no:
                filter_score[id][img_idx] = 0

    sum_of_filter_score = [0] * len(split_img_list)
    for question_id in range(len(filter_score)):
        for img_idx in range(len(split_img_list)):
            sum_of_filter_score[img_idx] += filter_score[question_id + 1][img_idx]
    
    sum_of_filter_score = [img_score / len(filter_score) for img_score in sum_of_filter_score]
    
    return sum(sum_of_filter_score)  / len(sum_of_filter_score) 
    
def main():
    args = parse_args()
    
    # Initialize distributed (multi-node/multi-GPU) environment if available
    def _dist_init_if_needed():
        if dist.is_available():
            if not dist.is_initialized():
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

    question_dependency_dir = "scripts/alignment"
    
    alignment_score_csv = f"results/alignment_score_{args.mode}_{formatted_time}.csv"
    alignment_prompt_score_csv = f"results/alignment_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(alignment_score_csv), exist_ok=True)

    # Configure per-rank cache roots to avoid shared-FS mmap/download conflicts
    base_cache = os.path.join(tempfile.gettempdir(), f"oneig_cache_rank{rank}")
    os.makedirs(base_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(base_cache, "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(base_cache, "hf_hub_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base_cache, "transformers_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base_cache, "xdg_cache"))

    # save the alignment score of each method (only used on rank 0 after aggregation)
    score_csv = pd.DataFrame(index=args.model_names, columns=["alignment"]) if rank == 0 else None
    
    # Instantiate inferencer with rank-0 first to avoid concurrent downloads
    # Decide warm-up leader: if cache lives inside repo (shared), use global rank 0; else per-node GPU0
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
            inferencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
        dist.barrier()
        if not is_leader:
            inferencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
    else:
        inferencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct", device=device)
    
    # Local collection of results as tuples: (row_key, model_name, result)
    local_results = []

    # Build and shard tasks by global index across ranks
    task_idx = 0
    for class_item in args.class_items:
        if rank == 0:
            print(f"We process {class_item} now.")

        if args.mode == "EN":
            question_dependency_json_dir = question_dependency_dir + '/Q_D/' + class_item + '.json'
        else:
            question_dependency_json_dir = question_dependency_dir + '/Q_D/' + class_item + '_zh.json'
 
        with open(question_dependency_json_dir, "r", encoding="utf-8") as f:
            question_dependency = json.load(f)
        
        # Ensure deterministic ordering across ranks
        keys = list(question_dependency.keys())
        try:
            keys = sorted(keys, key=lambda x: int(x))
        except Exception:
            keys = sorted(keys, key=lambda x: str(x))

        for key in tqdm(keys, desc=f"Processing {class_item}", disable=(rank != 0)):
            item = question_dependency[key]

            if isinstance(item["question"], str):
                item["question"] = {int(k): v for k, v in json.loads(item["question"]).items()}
            if isinstance(item["dependency"], str):
                item["dependency"] = {int(k): v for k, v in json.loads(item["dependency"]).items()}

            for model_id, model_name in enumerate(args.model_names):
                
                img_grid = (args.image_grid[model_id], args.image_grid[model_id])
                 
                image_path = megfile.smart_glob(args.image_dirname + '/' + class_item + '/' + model_name + '/' + key + '*')
                
                # Shard by task index across ranks
                if task_idx % world_size == rank:
                    result = alignment_score(image_path, item["question"], item["dependency"], img_grid, cache_dir, inferencer)
                    local_results.append((f"{class_item}_{key}", model_name, result))
                task_idx += 1

    # Aggregate results on rank 0
    if ddp_active:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
    else:
        gathered = [local_results]

    if rank == 0:
        # Flatten and build DataFrame
        all_results = []
        for part in gathered:
            all_results.extend(part)
        # Create prompt-level DataFrame
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for row_key, model_name, result in all_results:
            score_of_prompt_csv.loc[row_key, model_name] = result

        mean_values = score_of_prompt_csv.mean()
        score_csv["alignment"] = mean_values.values
        save2csv(score_csv, alignment_score_csv)
        # Optionally save per-prompt scores
        # score_of_prompt_csv = score_of_prompt_csv.sort_index()
        # save2csv(score_of_prompt_csv, alignment_prompt_score_csv)

        # Print parseable final results on rank 0
        result_dict = {}
        for model_name in args.model_names:
            row = score_csv.loc[model_name].to_dict()
            row = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            result_dict[model_name] = row
        print("FINAL_RESULT " + json.dumps({
            "script": "alignment",
            "mode": args.mode,
            "timestamp": formatted_time,
            "results": result_dict
        }))

    # Ensure all ranks reach this point before cleanup
    if ddp_active:
        dist.barrier()

    # Per-rank cache cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)
        
if __name__ == "__main__":
    main()