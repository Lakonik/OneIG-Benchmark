from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

import torchvision
torchvision.disable_beta_transforms_warning()
import tempfile
import torch
import torch.distributed as dist

import time
import socket

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def img_similar_score(image_1_path, image_2_path, model, preprocess, device):
    image_1 = preprocess(Image.open(image_1_path)).to(device)
    image_2 = preprocess(Image.open(image_2_path)).to(device)
    distance = model(image_1, image_2)
    return distance.item()

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
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    cache_dir = os.path.join(tempfile.gettempdir(), f"oneigbench_tmp_{formatted_time}_rank{rank}")
    os.makedirs(cache_dir, exist_ok=True)

    # Configure per-rank local caches to avoid network-FS mmap issues
    base_cache = os.path.join(tempfile.gettempdir(), f"oneig_cache_rank{rank}")
    os.makedirs(base_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(base_cache, "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(base_cache, "hf_hub_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base_cache, "transformers_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base_cache, "xdg_cache"))

    diversity_score_csv = f"results/diversity_score_{args.mode}_{formatted_time}.csv"
    diversity_prompt_score_csv = f"results/diversity_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(diversity_score_csv), exist_ok=True)

    # Initialize model per-rank with rank-0 warm-up to avoid concurrent downloads
    # Initialize DreamSim with temp working directory for models to avoid shared-FS races
    # Use a node-local directory (not NFS) for DreamSim to avoid safetensors mmap issues
    node_id = socket.gethostname()
    shared_models_dir = os.path.join(tempfile.gettempdir(), f"oneig_node_{node_id}", "dreamsim")
    if ddp_active:
        if local_rank == 0:
            os.makedirs(shared_models_dir, exist_ok=True)
        dist.barrier()

    old_cwd_local = os.getcwd()
    try:
        os.chdir(shared_models_dir)
        # Delay imports that might touch CUDA until after device/env is set
        from dreamsim import dreamsim
        from dreamsim.feature_extraction.extractor import ViTExtractor
        from dreamsim.feature_extraction.load_open_clip_as_dino import load_open_clip_as_dino
        if ddp_active:
            if local_rank == 0:
                model, preprocess = dreamsim(pretrained=True, device=device)
            dist.barrier()
            if local_rank != 0:
                model, preprocess = dreamsim(pretrained=True, device=device)
        else:
            model, preprocess = dreamsim(pretrained=True, device=device)
    finally:
        os.chdir(old_cwd_local)

    # Local results for per-prompt scores: list of (row_key, model_name, avg_score)
    local_results = []

    for model_id, model_name in enumerate(args.model_names):
        
        if rank == 0:
            print(f"It is {model_name} time.")
        
        img_grid = (args.image_grid[model_id], args.image_grid[model_id]) 

        # task sharding across images
        task_idx = 0

        for class_item in args.class_items:
            
            if rank == 0:
                print(f"We process {class_item} now.")
            
            image_dir = args.image_dirname + '/' + class_item + '/' + model_name
            img_list = megfile.smart_glob(image_dir + '/*')
            img_list = sorted(img_list)
            
            if rank == 0:
                print(f"We fetch {len(img_list)} images.")
            
            for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images", disable=(rank != 0)):
                
                if task_idx % (world_size if world_size > 0 else 1) == rank:
                    split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
                    if len(split_img_list) <= 1:
                        task_idx += 1
                        continue
                    score = []
                    for i in range(len(split_img_list)):
                        for j in range(i+1, len(split_img_list)):
                            prob = img_similar_score(split_img_list[i], split_img_list[j], model, preprocess, device)
                            score.append(prob)
                    avg_score = sum(score)/len(score)
                    local_results.append((f"{class_item}_{img_path.split('/')[-1][:3]}", model_name, class_item, avg_score))
                task_idx += 1

    # Gather results
    if ddp_active:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
    else:
        gathered = [local_results]

    if rank == 0:
        # Build DataFrames
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for part in gathered:
            for row_key, model_name, class_item, avg_score in part:
                score_of_prompt_csv.loc[row_key, model_name] = avg_score
        # Per-class averages per model
        score_csv = pd.DataFrame(index=args.model_names, columns=args.class_items + ["total average"])
        for model_name in args.model_names:
            for class_item in args.class_items:
                # rows that belong to this class
                class_rows = [idx for idx in score_of_prompt_csv.index if idx.startswith(f"{class_item}_")]
                if len(class_rows) > 0:
                    vals = score_of_prompt_csv.loc[class_rows, model_name].dropna()
                    score_csv.loc[model_name, class_item] = vals.mean() if len(vals) > 0 else None
                else:
                    score_csv.loc[model_name, class_item] = None
        mean_values = score_of_prompt_csv.mean()
        score_csv["total average"] = mean_values.values
        save2csv(score_csv, diversity_score_csv)
        # Optionally save prompt-level scores
        # score_of_prompt_csv = score_of_prompt_csv.sort_index()
        # save2csv(score_of_prompt_csv, diversity_prompt_score_csv)

    if ddp_active:
        dist.barrier()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()