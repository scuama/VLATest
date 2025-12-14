#!/usr/bin/env python3
"""
Run a single test entry through OpenVLA -> ManiSkill pipeline.

Usage (example):
  python experiments/run_one_openvla_example.py \
    --data ../data/t-grasp_n-100_o-0_s-170912623.json \
    --idx 0 \
    --model openvla-7b

This script is a minimal wrapper around `experiments.model_interface.VLAInterface`.
It loads one `options` entry from the JSON, creates the environment, runs the model
until termination, and saves images + a small log.json in the output folder.

Notes:
- Requires the simulator and model dependencies from the repo (mani_skill2_real2sim, simpler_env, PyTorch/Transformers, etc.).
- If you use a finetuned LoRA adapter, pass `--lora_path /path/to/lora`.
"""
import argparse
import json
import os
from pathlib import Path
from experiments.model_interface import VLAInterface
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', type=str, required=True, help='Path to dataset JSON')
    p.add_argument('--idx', type=int, default=0, help='Index of test case in JSON')
    p.add_argument('--model', '-m', type=str, default='openvla-7b', help='Model name')
    p.add_argument('--lora_path', type=str, default=None, help='Optional LoRA adapter path')
    p.add_argument('--seed', type=int, default=2024, help='Random seed for env.reset')
    p.add_argument('--output', '-o', type=str, default=None, help='Output directory prefix')
    args = p.parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"Data file not found: {data_path}"

    with open(data_path, 'r') as f:
        tasks = json.load(f)

    idx = args.idx
    key = str(idx)
    assert key in tasks, f"Index {idx} not found in {data_path}"
    options = tasks[key]

    # Infer task name from filename (same logic as run_fuzzer)
    dataset_name = data_path.name
    if "grasp" in dataset_name:
        task = "google_robot_pick_customizable_ycb" if 'ycb' in dataset_name else "google_robot_pick_customizable"
    elif "move" in dataset_name:
        task = "google_robot_move_near_customizable_ycb" if 'ycb' in dataset_name else "google_robot_move_near_customizable"
    elif "put-on" in dataset_name:
        task = "widowx_put_on_customizable_ycb" if 'ycb' in dataset_name else "widowx_put_on_customizable"
    elif "put-in" in dataset_name:
        task = "widowx_put_in_customizable_ycb" if 'ycb' in dataset_name else "widowx_put_in_customizable"
    else:
        raise NotImplementedError(f"Cannot infer task from dataset name: {dataset_name}")

    # Create VLA interface
    vla = VLAInterface(task=task, model_name=args.model, lora_path=args.lora_path)

    images, episode_stats = vla.run_interface(seed=args.seed, options=options)

    # Output folder
    out_base = args.output if args.output else (Path(__file__).parent / 'outputs')
    out_dir = Path(out_base) / f"{data_path.stem}_{args.model}_{args.seed}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save log
    with open(out_dir / 'log.json', 'w') as f:
        json.dump(episode_stats, f)

    # Save images (if any)
    for i, im in enumerate(images):
        Image.fromarray(im).save(out_dir / f'{i:03d}.jpg')

    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()
