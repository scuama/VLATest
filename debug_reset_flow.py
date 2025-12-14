#!/usr/bin/env python3
"""调试reset流程，添加日志跟踪"""
import json
import sys
import numpy as np

# 添加路径
sys.path.insert(0, '/mnt/disk1/decom/VLATest')

# Monkey patch 关键函数来跟踪调用
original_load_model = None
original_clear_sim_state = None
original_reconfigure = None

def track_load_model(self):
    print("  🔧 _load_model() 被调用 - 重新加载物体模型")
    return original_load_model(self)

def track_clear_sim_state(self):
    print("  🧹 _clear_sim_state() 被调用 - 仅清除速度状态")
    return original_clear_sim_state(self)

def track_reconfigure(self, options):
    print("  🔄 reconfigure() 被调用 - 完全重建场景!!!")
    return original_reconfigure(self, options)

# 应用monkey patch
import simpler_env
from mani_skill2_real2sim.envs.sapien_env import BaseEnv

original_reconfigure = BaseEnv.reconfigure
original_clear_sim_state = BaseEnv._clear_sim_state
BaseEnv.reconfigure = track_reconfigure
BaseEnv._clear_sim_state = track_clear_sim_state

print("=" * 70)
print("测试1: 原始配置（无robot_init_options）")
print("=" * 70)

# 原始配置
with open('/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37_backup/options.json') as f:
    options1 = json.load(f)

print(f"\n配置内容: {list(options1.keys())}")
print(f"有robot_init_options: {'robot_init_options' in options1}")

env1 = simpler_env.make('google_robot_move_near_customizable')
print("\n开始reset...")
obs1, info1 = env1.reset(seed=options1.get('seed'), options=options1)
print(f"Reset完成\n")

# 检查物体是否加载
try:
    objs = env1.unwrapped.episode_objs
    print(f"✅ 加载了 {len(objs)} 个物体")
    for obj in objs:
        print(f"   - {obj.name}: 位置 {obj.pose.p[:2]}")
except Exception as e:
    print(f"❌ 获取物体失败: {e}")

env1.close()

print("\n" + "=" * 70)
print("测试2: 修改配置（有robot_init_options）")
print("=" * 70)

# 修改后的配置
with open('/mnt/disk1/decom/VLATest/test_ep37_modified/options.json') as f:
    options2 = json.load(f)

print(f"\n配置内容: {list(options2.keys())}")
print(f"有robot_init_options: {'robot_init_options' in options2}")
if 'robot_init_options' in options2:
    print(f"robot_init_options = {options2['robot_init_options']}")

env2 = simpler_env.make('google_robot_move_near_customizable')
print("\n开始reset...")
obs2, info2 = env2.reset(seed=options2.get('seed'), options=options2)
print(f"Reset完成\n")

# 检查物体是否加载
try:
    objs = env2.unwrapped.episode_objs
    print(f"✅ 加载了 {len(objs)} 个物体")
    for obj in objs:
        print(f"   - {obj.name}: 位置 {obj.pose.p[:2]}")
except Exception as e:
    print(f"❌ 获取物体失败: {e}")

env2.close()
