from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

import torch
torch.cuda.empty_cache()
from scripts.utils.inference import CSDStyleEmbedding, SEStyleEmbedding
import tempfile
import torch.distributed as dist
import socket

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

style_list = ['abstract_expressionism', 'art_nouveau', 'baroque', 'chinese_ink_painting', 'cubism', 'fauvism', 'impressionism', 'line_art', 'minimalism', 'pointillism', 'pop_art', 'rococo',  'ukiyo-e', 'clay', 'crayon',  'graffiti','lego', 'comic', 'pencil_sketch', 'stone_sculpture', 'watercolor', 'celluloid', 'chibi',   'cyberpunk',  'ghibli',  'impasto', 'pixar', 'pixel_art',  '3d_rendering']

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

    style_csv_path = "scripts/style/style.csv"
    df = pd.read_csv(style_csv_path, dtype=str)
    
    # Avoid concurrent downloads/loads (HF) by warming up on rank 0
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
            CSD_Encoder = CSDStyleEmbedding(model_path="scripts/style/models/checkpoint.pth", device=device)
            SE_Encoder = SEStyleEmbedding(pretrained_path="xingpng/OneIG-StyleEncoder", device=device)
        dist.barrier()
        if not is_leader:
            CSD_Encoder = CSDStyleEmbedding(model_path="scripts/style/models/checkpoint.pth", device=device)
            SE_Encoder = SEStyleEmbedding(pretrained_path="xingpng/OneIG-StyleEncoder", device=device)
    else:
        CSD_Encoder = CSDStyleEmbedding(model_path="scripts/style/models/checkpoint.pth", device=device)
        SE_Encoder = SEStyleEmbedding(pretrained_path="xingpng/OneIG-StyleEncoder", device=device)

    CSD_embed_pt = "scripts/style/CSD_embed.pt"
    CSD_ref = torch.load(CSD_embed_pt, weights_only=False, map_location=device)
    SE_embed_pt = "scripts/style/SE_embed.pt"
    SE_ref = torch.load(SE_embed_pt, map_location=device)

    style_score_csv = f"results/style_score_{args.mode}_{formatted_time}.csv"
    style_style_score_csv = f"results/style_style_score_{args.mode}_{formatted_time}.csv"
    style_prompt_score_csv = f"results/style_prompt_score_{args.mode}_{formatted_time}.csv"
    if rank == 0:
        os.makedirs(os.path.dirname(style_score_csv), exist_ok=True)

    score_csv = pd.DataFrame(index=args.model_names, columns=["style"])
    score_of_style_csv = pd.DataFrame(index=args.model_names, columns=style_list)
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)  
    
    # Local results: (id, model_name, avg_style_score, image_style)
    local_prompt_results = []
    # Per-style accumulators collected later on rank 0
    for model_id, model_name in enumerate(args.model_names):
        
        if rank == 0:
            print(f"It is {model_name} time.")
        
        img_grid = (args.image_grid[model_id], args.image_grid[model_id]) 
        
        image_dir = args.image_dirname + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        img_list = sorted(img_list)
        
        if rank == 0:
            print(f"We fetch {len(img_list)} images.")
        
        style_dict = {style: [] for style in style_list}

        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images", disable=(rank != 0)):
            if idx % (world_size if world_size > 0 else 1) != rank:
                continue
            
            id = img_path.split('/')[-1][:3]
            
            image_style =  str(df.loc[df["id"] == id, "class"].values[0])
            if (image_style[:3] == "nan"):
                continue
            else:
                image_style = image_style.lower().replace(' ', '_')
            
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)

            CSD_ref_embeds = CSD_ref[image_style].to(device)
            SE_ref_embeds = SE_ref[image_style].to(device)
            
            score = []
            for num, split_img_path in enumerate(split_img_list):
                
                CSD_embed = CSD_Encoder.get_style_embedding(split_img_path)
                SE_embed = SE_Encoder.get_style_embedding(split_img_path)
                
                CSD_max_style_score = max(torch.max(CSD_embed @ CSD_ref_embeds.T).item(), 0)
                SE_max_style_score = max(torch.max(SE_embed @ SE_ref_embeds.T).item(), 0)
                
                max_style_score = (CSD_max_style_score + SE_max_style_score) / 2
                score.append(max_style_score)
            
            if len(score) != 0:
                avg_val = sum(score)/len(score)
                local_prompt_results.append((id, model_name, avg_val, image_style))
                style_dict[image_style].append(avg_val)
            else:
                local_prompt_results.append((id, model_name, None, image_style))        
                    
        # We keep per-rank style_dict; aggregation will happen after gather on rank 0

    # Gather results across ranks
    if ddp_active:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_prompt_results, gathered, dst=0)
    else:
        gathered = [local_prompt_results]

    if rank == 0:
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)  
        # style per-model aggregator
        per_model_style_vals = {m: {s: [] for s in style_list} for m in args.model_names}
        for part in gathered:
            for id, model_name, avg_val, image_style in part:
                score_of_prompt_csv.loc[id, model_name] = avg_val
                if avg_val is not None:
                    per_model_style_vals[model_name][image_style].append(avg_val)
        for model_name in args.model_names:
            for style in style_list:
                vals = per_model_style_vals[model_name][style]
                if len(vals) > 0:
                    score_of_style_csv.loc[model_name, style] = sum(vals) / len(vals)

        mean_values = score_of_prompt_csv.mean()
        score_csv["style"] = mean_values.values
        save2csv(score_csv, style_score_csv)
    
    # save2csv(score_of_style_csv, style_style_score_csv)

    # score_of_prompt_csv = score_of_prompt_csv.sort_index()
    # save2csv(score_of_prompt_csv, style_prompt_score_csv)    

    if ddp_active:
        dist.barrier()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
