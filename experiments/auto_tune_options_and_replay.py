import os
import json
import random
import subprocess
import numpy as np

# ====== 你需要修改的部分 ======
EPISODE_DIR = "/mnt/disk1/decom/VLATest/newresult/t-grasp_n-100_o-0_s-170912623-2-0/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623-2/openvla-7b_2024/0"
TASK_NAME = "google_robot_pick_customizable"
REPLAY_SCRIPT_PATH = "/mnt/disk1/decom/VLATest/experiments/replay_openvla_actions.py"   # 或绝对路径
# =================================


def load_options(path):
    file = os.path.join(path, "options.json")
    with open(file, "r") as f:
        return json.load(f)


def save_options(path, data):
    file = os.path.join(path, "options.json")
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def random_unit_quaternion():
    """随机生成正规化四元数 (w,x,y,z)"""
    u1, u2, u3 = np.random.rand(3)
    q = [
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ]
    return q


def random_modify_options(opts):
    """
    针对你的 options.json 格式定制的随机扰动策略
    """
    # 随机物体初始位置（平面）
    opts["obj_init_options"]["init_xy"][0] = round(random.uniform(-0.35, 0.35), 3)
    opts["obj_init_options"]["init_xy"][1] = round(random.uniform(-0.35, 0.35), 3)

    # 随机物体朝向（单位四元数）
    opts["obj_init_options"]["orientation"] = random_unit_quaternion()

    return opts


def run_replay():
    """执行重放脚本"""
    cmd = f"python3 {REPLAY_SCRIPT_PATH} --episode_dir {EPISODE_DIR} --task {TASK_NAME} --render_every 1"
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_success(episode_dir):
    """
    判断重放是否成功。
    success: true 需要出现在 replay_log.json 的 info 字段里。
    """
    log_file = os.path.join(episode_dir, "replay_log.json")
    if not os.path.exists(log_file):
        return False

    with open(log_file, "r") as f:
        logs = json.load(f)

    for step, info in logs.items():
        if isinstance(info, dict) and info.get("success") is True:
            return True

    return False


def main():
    print("===== 自动随机调参 + 重放开始 =====\n")

    attempt = 0
    while True:
        attempt += 1
        print(f"\n===== 第 {attempt} 次尝试 =====")

        # 重新加载 options.json
        opts = load_options(EPISODE_DIR)

        # 随机修改
        opts = random_modify_options(opts)

        # 保存新的 options.json
        save_options(EPISODE_DIR, opts)
        print("当前随机 options.json：")
        print(json.dumps(opts, indent=2))

        # 执行重放
        run_replay()

        # 检查成功
        if check_success(EPISODE_DIR):
            print("\n🎉🎉🎉 找到了成功的参数配置！")
            print(json.dumps(opts, indent=2))
            break

        print("失败，继续随机尝试...")


if __name__ == "__main__":
    main()

#python3 auto_tune_options_and_replay.py