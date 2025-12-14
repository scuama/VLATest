#!/usr/bin/env python3
"""测试：检查urdf_version随机选择机制"""
import numpy as np

# 模拟原始reset（无robot_init_options）
print("=" * 70)
print("场景1: 原始reset（无robot_init_options指定）")
print("=" * 70)

# 设置相同的seed
seed = 3225323079
rng = np.random.RandomState(seed)

# 跳过37次选择（因为是37号episode）
for i in range(37):
    _ = rng.randint(2**32)  # episode seed

# 第37个episode的rng
episode_seed = rng.randint(2**32)
episode_rng = np.random.RandomState(episode_seed)
print(f"Episode 37 seed: {episode_seed}")

# _additional_prepackaged_config_reset 中的urdf_version选择
choices = [
    "",
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    "recolor_cabinet_visual_matching_1",
]
selected_urdf = episode_rng.choice(choices)
print(f"选择的urdf_version: '{selected_urdf}'")
print(f"会触发reconfigure: {selected_urdf != ''}")
print()

# 模拟修改后reset（有robot_init_options）
print("=" * 70)
print("场景2: 修改后reset（有robot_init_options指定）")
print("=" * 70)

# 重置相同的seed
rng = np.random.RandomState(seed)
for i in range(37):
    _ = rng.randint(2**32)

episode_seed = rng.randint(2**32)
episode_rng = np.random.RandomState(episode_seed)
print(f"Episode 37 seed: {episode_seed}")

# _additional_prepackaged_config_reset 中的逻辑相同
selected_urdf = episode_rng.choice(choices)
print(f"选择的urdf_version: '{selected_urdf}'")
print(f"会触发reconfigure: {selected_urdf != ''}")
