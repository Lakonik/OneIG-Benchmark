from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error, setup_distributed

import torchvision
torchvision.disable_beta_transforms_warning()
import tempfile
import torch
import torch.distributed as dist

import time
import socket
import json

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

    ddp_active, world_size, rank, local_rank, device = setup_distributed()

    cache_dir = os.path.join(tempfile.gettempdir(), f"oneigbench_tmp_{formatted_time}_rank{rank}")
    os.makedirs(cache_dir, exist_ok=True)


    diversity_score_csv = f"results/diversity_score_{args.mode}_{formatted_time}.csv"
    diversity_prompt_score_csv = f"results/diversity_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(diversity_score_csv), exist_ok=True)

    # Initialize model per-rank with rank-0 warm-up to avoid concurrent downloads
    # Initialize DreamSim with temp working directory for models to avoid shared-FS races
    # Use a node-local directory (not NFS) for DreamSim to avoid safetensors mmap issues
    node_id = socket.gethostname()
    shared_models_dir = os.path.join(tempfile.gettempdir(), f"oneig_node_{node_id}", "dreamsim")
    os.makedirs(shared_models_dir, exist_ok=True)

    old_cwd_local = os.getcwd()
    try:
        os.chdir(shared_models_dir)
        # Delay imports that might touch CUDA until after device/env is set
        from dreamsim import dreamsim
        from dreamsim.feature_extraction.extractor import ViTExtractor
        from dreamsim.feature_extraction.load_open_clip_as_dino import load_open_clip_as_dino
        # DreamSim has no download locking of its own: on a cold cache every local rank
        # would fetch the same pretrained.zip and run extractall() concurrently. A file
        # lock (same idiom as lakonlab's locked_cache_path) serializes the first
        # populate-and-load per node; later holders see the ready marker and just load.
        from huggingface_hub.utils import WeakFileLock
        ready_marker = os.path.join(shared_models_dir, ".dreamsim_ready")
        with WeakFileLock(os.path.join(shared_models_dir, ".dreamsim.lock")):
            if not os.path.exists(ready_marker):
                for name in os.listdir(shared_models_dir):  # wipe any partial download
                    if name.startswith("."):
                        continue
                    full = os.path.join(shared_models_dir, name)
                    shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) else os.remove(full)
            model, preprocess = dreamsim(pretrained=True, device=device)
            if not os.path.exists(ready_marker):
                open(ready_marker, "w").close()
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

        # Print parseable final results on rank 0
        result_dict = {}
        for model_name in args.model_names:
            row = score_csv.loc[model_name].to_dict()
            row = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            result_dict[model_name] = row
        print("FINAL_RESULT " + json.dumps({
            "script": "diversity",
            "mode": args.mode,
            "timestamp": formatted_time,
            "results": result_dict
        }))

    if ddp_active:
        dist.barrier()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()