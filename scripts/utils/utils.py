import os
import stat
import megfile
import argparse
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def setup_distributed():
    """Initialize distributed state and report this process's device.

    Standalone under a launcher (RANK/WORLD_SIZE in the environment): the process is
    bound to its LOCAL_RANK GPU *before* NCCL initialization (torchrun does not call
    set_device; without the bind every process lands on cuda:0), and
    init_process_group failures raise instead of silently degrading into world-size 1,
    where every process would evaluate the full dataset and race on the outputs.
    Binding is skipped when each rank only sees a single GPU (LOCAL_RANK >=
    device_count), where logical device 0 is already correct.

    Embedded callers (evaluation inside a training process) arrive with dist already
    initialized and the device already selected by the trainer; nothing is touched.

    Without launcher env vars this is a plain single-process run and no init is
    attempted.
    """
    import torch
    import torch.distributed as dist

    pre_initialized = dist.is_available() and dist.is_initialized()
    if not pre_initialized and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if torch.cuda.is_available():
            launch_local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if launch_local_rank < torch.cuda.device_count():
                torch.cuda.set_device(launch_local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    ddp_active = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if ddp_active else 1
    rank = dist.get_rank() if ddp_active else 0
    local_rank = dist.get_node_local_rank() if ddp_active else 0

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    return ddp_active, world_size, rank, local_rank, device


def parse_args():
    parser = argparse.ArgumentParser(description="Run alignment score evaluation.")
    parser.add_argument("--mode", type=str, default="EN", help="Choose language mode.")
    parser.add_argument("--image_dirname", type=str, default="images", help="Directory containing images.")
    parser.add_argument("--model_names", type=str, nargs="+", default=["gpt-4o"], help="List of model names.")
    parser.add_argument("--image_grid", type=int, nargs="+", default=[2], help="List of image grids.")
    parser.add_argument("--class_items", type=str, nargs="+", default=["anime", "human", "object"], help="List of class items.")
    return parser.parse_args()

def is_black_image(image):
    pixels = image.load()
    for i in range(image.width):
        for j in range(image.height):
            if pixels[i, j] != (0, 0, 0):
                return False
    return True

def split_2x2_grid(image_path, grid_size, cache_dir):
    with megfile.smart_open(image_path, 'rb') as f:
        grid_image = Image.open(f)

        width, height = grid_image.size

        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list = []

        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,
                    i * individual_height,
                    (j + 1) * individual_width,
                    (i + 1) * individual_height
                )

                individual_image = grid_image.crop(box)

                if is_black_image(individual_image):
                    print(f"Detected a black image at position ({i},{j}) in {image_path}")
                else:
                    image_list.append(individual_image)

    image_path_list = []
    for i, image in enumerate(image_list):
        image_path = os.path.join(cache_dir, f"{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)

    return image_path_list

def save2csv(df, csv_path):
    df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
