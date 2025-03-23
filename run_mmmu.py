"""
Script to run a VLM model on the MMMU Geography dataset.
"""

import wandb
import argparse
import threading
import time
import warnings
from ast import literal_eval

import torch
import torchvision.transforms as T
from datasets import load_dataset
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

wandb.init(project="multimodal-abc")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

generation_config = {"max_new_tokens": 1024, "do_sample": True}


def load_mmmu_data(batch_size: int):
    """Load MMMU dataset from huggingface."""
    data = load_dataset("MMMU/MMMU", "Geography", split="test")
    data = data.filter(lambda x: x["image_1"] is not None)
    data = data.filter(lambda x: x["image_2"] is None)
    data = data.select(range(batch_size))
    return data


def load_artifacts(model_name: str):
    """Load model and tokenizer from huggingface."""
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


def build_transform(input_size):
    """Image Transforms"""
    transform = T.Compose(
        [
            T.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio, target_ratios, width, height, image_size
):
    """Function to find the closest aspect ratio to the target."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(  # pylint: disable=too-many-locals
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """Function to dynamically preprocess the image."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """Load image and preprocess."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def gpu_memory_monitor(interval=0.1, gpu_id=0):
    """
    Monitor GPU memory usage in a separate thread.

    Args:
        interval (float): Sampling interval in seconds
        gpu_id (int): ID of the GPU to monitor

    Returns:
        start_monitor, stop_monitor functions and a results dictionary
    """
    nvmlInit()
    results = {"used_memory": [], "total_memory": [], "is_running": False}

    monitor_thread = None

    def memory_monitoring_task():
        handle = nvmlDeviceGetHandleByIndex(gpu_id)
        while results["is_running"]:
            info = nvmlDeviceGetMemoryInfo(handle)
            results["used_memory"].append(info.used / 1024**2)  # Convert to MB
            results["total_memory"].append(info.total / 1024**2)
            time.sleep(interval)

    def start_monitor():
        results["is_running"] = True
        nonlocal monitor_thread
        monitor_thread = threading.Thread(target=memory_monitoring_task)
        monitor_thread.daemon = True
        monitor_thread.start()

    def stop_monitor():
        results["is_running"] = False
        if monitor_thread:
            monitor_thread.join(timeout=2 * interval)

        # Calculate and print maximum memory usage
        if results["used_memory"]:
            max_memory = max(results["used_memory"])
            total_memory = results["total_memory"][0]
            print("\nMaximum GPU Memory Usage:")
            print(f"Used: {max_memory:.2f} MB")
            print(f"Total: {total_memory:.2f} MB")
            print(f"Percentage: {(max_memory/total_memory)*100:.2f}%")
            log = {
                "max_memory": max_memory,
                "total_memory": total_memory,
                "percentage": (max_memory / total_memory) * 100,
            }
            wandb.log(log)
            wandb.finish()
        return results

    return start_monitor, stop_monitor, results


def monitor_gpu_usage(func):
    """Decorator to monitor GPU usage while a function runs"""

    def wrapper(*args, **kwargs):
        start_monitor, stop_monitor, results = (
            gpu_memory_monitor()
        )  # pylint: disable=unused-variable
        start_monitor()
        try:
            return_value = func(*args, **kwargs)
        finally:
            stop_monitor()
        return return_value

    return wrapper


def main():
    """Program's entrypoint"""
    parser = argparse.ArgumentParser(description="Run VLM on MMMU Geography")
    parser.add_argument(
        "--model_name", default="OpenGVLab/InternVL2_5-1B", type=str
    )
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_artifacts(args.model_name)

    @monitor_gpu_usage
    def process_batch(batch):
        # Process text for all rows in batch
        texts = []
        for q, opts in zip(batch["question"], batch["options"]):
            if len(opts) != 0:
                texts.append(
                    q + "\n\noptions:\n- " + "\n- ".join(literal_eval(opts))
                )
            else:
                texts.append(q)

        # Process images for all rows in batch
        pixel_values = []
        num_patches = []
        for idx in range(len(batch["question"])):  # iterate over batch items
            img = batch["image_1"][idx]
            row_pixel_value = (
                load_image(img, max_num=12).to(torch.bfloat16).cuda()
            )
            pixel_values.append(row_pixel_value)
            num_patches.append(row_pixel_value.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)
        model.to("cuda")
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches,
            questions=texts,
            generation_config=generation_config,
        )
        batch["response"] = responses
        del pixel_values
        torch.cuda.empty_cache()
        return batch

    # Load data
    print("Loading data...")
    ds = load_mmmu_data(args.batch_size)

    # Run model on data
    ds = ds.map(process_batch, batched=True, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
