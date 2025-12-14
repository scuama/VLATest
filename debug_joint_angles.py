#!/usr/bin/env python3
"""检查机械臂关节角度"""
import json
import sys
import numpy as np

sys.path.insert(0, '/mnt/disk1/decom/VLATest')
import simpler_env

def check_robot_state(options, label):
    print("=" * 70)
    print(f"{label}")
    print("=" * 70)
    
    env = simpler_env.make('google_robot_move_near_customizable')
    obs, info = env.reset(seed=options.get('seed'), options=options)
    
    robot = env.unwrapped.agent.robot
    
    print(f"\n机械臂基座位置: {robot.pose.p}")
    print(f"机械臂基座四元数: {robot.pose.q}")
    
    # 获取关节状态
    qpos = robot.get_qpos()
    print(f"\n关节位置 (qpos): {qpos}")
    
    # 获取各连杆位置
    print(f"\n主要连杆位置:")
    for link in robot.get_links():
        if 'link' in link.name or 'camera' in link.name:
            print(f"  {link.name:20s}: 位置{link.pose.p} 四元数{link.pose.q}")
    
    env.close()
    print()

# 测试两种配置
with open('/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37_backup/options.json') as f:
    options1 = json.load(f)
check_robot_state(options1, "原始配置（无robot_init_options）")

with open('/mnt/disk1/decom/VLATest/test_ep37_modified/options.json') as f:
    options2 = json.load(f)
check_robot_state(options2, "修改后配置（有robot_init_options）")
