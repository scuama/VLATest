#!/usr/bin/env python3
"""调试相机视角问题"""
import json
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/mnt/disk1/decom/VLATest')
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

def test_camera_view(options, label):
    print("=" * 70)
    print(f"测试: {label}")
    print("=" * 70)
    
    env = simpler_env.make('google_robot_move_near_customizable')
    obs, info = env.reset(seed=options.get('seed'), options=options)
    
    # 获取机械臂位置
    robot_pose = env.unwrapped.agent.robot.pose
    print(f"\n🤖 机械臂位置: [{robot_pose.p[0]:.4f}, {robot_pose.p[1]:.4f}, {robot_pose.p[2]:.4f}]")
    
    # 获取物体位置
    print(f"\n📦 物体位置:")
    for obj in env.unwrapped.episode_objs:
        print(f"   {obj.name}: [{obj.pose.p[0]:.4f}, {obj.pose.p[1]:.4f}, {obj.pose.p[2]:.4f}]")
    
    # 获取相机位置（挂载在link_camera上）
    try:
        link_camera = None
        for link in env.unwrapped.agent.robot.get_links():
            if link.name == "link_camera":
                link_camera = link
                break
        
        if link_camera:
            camera_pose = link_camera.pose
            print(f"\n📷 相机连杆位置: [{camera_pose.p[0]:.4f}, {camera_pose.p[1]:.4f}, {camera_pose.p[2]:.4f}]")
            print(f"   相机朝向: {camera_pose.q}")
        else:
            print(f"\n❌ 未找到link_camera连杆")
    except Exception as e:
        print(f"\n❌ 获取相机信息失败: {e}")
    
    # 渲染图像
    try:
        img = get_image_from_maniskill2_obs_dict(env, obs)
        if img is not None:
            # 保存图像
            output_path = f'/mnt/disk1/decom/VLATest/debug_{label.replace(" ", "_")}.jpg'
            Image.fromarray(img).save(output_path)
            print(f"\n✅ 图像已保存: {output_path}")
            print(f"   图像尺寸: {img.shape}")
            
            # 简单检查图像内容（非空白）
            mean_val = img.mean()
            print(f"   图像平均像素值: {mean_val:.2f}")
        else:
            print(f"\n❌ 渲染失败，图像为None")
    except Exception as e:
        print(f"\n❌ 渲染失败: {e}")
    
    env.close()
    print()

# 测试1: 原始配置
with open('/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37_backup/options.json') as f:
    options1 = json.load(f)
test_camera_view(options1, "原始配置")

# 测试2: 修改后配置
with open('/mnt/disk1/decom/VLATest/test_ep37_modified/options.json') as f:
    options2 = json.load(f)
test_camera_view(options2, "修改后配置")
