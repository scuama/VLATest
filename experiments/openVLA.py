"""
Name   : run_fuzzer.py
Author : ZHIJIE WANG
Time   : 8/8/24
"""
import argparse
import numpy as np
from model_interface import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
from PIL import Image
import shutil

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()
PROMPT_TEMPLATES = {
     "template": "{instruction}",
}
class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-d', '--data', type=str, help="Testing data")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output path, e.g., folder")
    parser.add_argument('-io', '--image_output', type=str, default=None, help="Image output path, e.g., folder")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-m', '--model', type=str,
                        choices=["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b"],
                        default="openvla-7b",
                        help="VLA model")
    parser.add_argument('-l', '--lora_path', type=str, default=None,
                        help="LoRA adapter path for finetuned OpenVLA model")
    parser.add_argument('-r', '--resume', action='store_true', default=False, help="Resume from where we left.")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    data_path = args.data if args.data else str(PACKAGE_DIR) + "/../data/t-grasp_n-1000_o-3.json"

    dataset_name = data_path.split('/')[-1]

    # test_small_10 默认为 grasp 任务
    if "grasp" in dataset_name or "test_small" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable_ycb", lora_path=args.lora_path)
        else:
            vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable", lora_path=args.lora_path)
    elif "move" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable_ycb", lora_path=args.lora_path)
        else:
            vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable", lora_path=args.lora_path)
    elif "put-on" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable_ycb", lora_path=args.lora_path)
        else:
            vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable", lora_path=args.lora_path)
    elif "put-in" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable_ycb", lora_path=args.lora_path)
        else:
            vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable", lora_path=args.lora_path)
    else:
        vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable_ycb", lora_path=args.lora_path)

    with open(data_path, 'r') as f:
        tasks = json.load(f)
    
    # 兼容有无 "num" 字段的数据格式
    if "num" not in tasks:
        tasks["num"] = len([k for k in tasks.keys() if k.isdigit()])
    
    print(args.output)
    
    if args.output:
        result_dir = args.output + data_path.split('/')[-1].split(".")[0]
    else:
        result_dir = str(PACKAGE_DIR) + "/../results/" + data_path.split('/')[-1].split(".")[0]
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)
    
    # 如果使用LoRA，在目录名中标注
    model_tag = f"{args.model}_finetuned" if args.lora_path else args.model
    result_dir += f'/{model_tag}_{random_seed}'
    if not args.resume:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    if args.output:
        image_dir = args.output + data_path.split('/')[-1].split(".")[0]+"/images"
        os.makedirs(image_dir, exist_ok=True)
        image_dir += f'/{args.model}_{random_seed}'
        os.makedirs(image_dir, exist_ok=True)
    else:
        image_dir = None
    # 获取所有 episode 键（排除 'num' 和 'seed'）
    episode_keys = [k for k in tasks.keys() if k not in ['num', 'seed']]
    
    for idx in tqdm(range(tasks["num"])):
       
        
        # 支持两种键格式：索引键（"0", "1", "2"）或 episode_id 键（"7", "2", "44"）
        if str(idx) in tasks:
            episode_key = str(idx)
        elif idx < len(episode_keys):
            episode_key = episode_keys[idx]
        else:
            print(f"⚠️ 跳过索引 {idx}：找不到对应的配置")
            continue
        
        if args.resume and os.path.exists(result_dir + f"/{episode_key}/" + '/log.json'):  # if resume allowed then skip the finished runs.

            continue
        options = tasks[episode_key]
        # 设置最大步数以加速测试（如果options中没有设置的话）
        if "max_episode_steps" not in options:
            options["max_episode_steps"] = 20  # 可以根据需要调整这个数值
      
        images, episode_stats, actions = vla.run_interfaceWithPromot(seed=random_seed, options=options,promot=PROMPT_TEMPLATES)
        os.makedirs(result_dir + f"/{episode_key}", exist_ok=True)
        with open(result_dir + f"/{episode_key}/" + '/log.json', "w") as f:
            json.dump(episode_stats, f, cls=StableJSONizer)
        if image_dir:
            os.makedirs(image_dir + f"/{episode_key}", exist_ok=True)
            for img_idx in range(len(images)):
                im = Image.fromarray(images[img_idx])
                im.save(image_dir + f"/{episode_key}/" + f'{img_idx}.jpg')
        # updated by zeqin: save actions and options for later replay in ManiSkill
        try:
            # try saving as numpy file first (allowing object dtype)
            np.save(result_dir + f"/{episode_key}/" + 'actions.npy', np.array(actions, dtype=object))
        except Exception:
            pass
        try:
            # also save as json-serializable list of lists (more portable)
            actions_list = [a.tolist() if hasattr(a, 'tolist') else None for a in actions]
            with open(result_dir + f"/{episode_key}/" + 'actions.json', 'w') as fa:
                json.dump(actions_list, fa)
        except Exception:
            pass
        try:
            with open(result_dir + f"/{episode_key}/" + 'options.json', 'w') as fo:
                json.dump(options, fo, cls=StableJSONizer)
        except Exception:
            try:
                with open(result_dir + f"/{episode_key}/" + 'options.json', 'w') as fo:
                    json.dump(options, fo)
            except Exception:
                pass