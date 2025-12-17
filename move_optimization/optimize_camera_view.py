#!/usr/bin/env python3
"""
优化相机视角以完整看到两个物体

策略：
1. 调整相机位置（后退、升高）以扩大视野
2. 调整相机FOV（视场角）以获得更广的视角
3. 调整相机朝向以同时覆盖两个物体
"""

import os
import sys
import json
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qinverse
import sapien.core as sapien

sys.path.insert(0, '/mnt/disk1/decom/VLATest')
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

def calculate_camera_params_for_two_objects(robot_xy, obj1_xy, obj2_xy):
    """
    计算能够同时看到两个物体的最佳相机参数
    
    Args:
        robot_xy: 机械臂位置 [x, y]
        obj1_xy: 物体1位置 [x, y]
        obj2_xy: 物体2位置 [x, y]
    
    Returns:
        dict: 相机优化参数 {
            'position_offset': [dx, dy, dz],  # 相对于默认位置的偏移
            'rotation_offset': [roll, pitch, yaw],  # 欧拉角偏移
            'fov_multiplier': float  # FOV缩放因子
        }
    """
    # 默认相机参数（从WidowXBridgeDatasetCameraSetupConfig）
    default_camera_pos = np.array([0.00, -0.16, 0.336])  # 相对于base_link
    
    # 计算两个物体的中心点（世界坐标系）
    obj_center = np.array([
        (obj1_xy[0] + obj2_xy[0]) / 2,
        (obj1_xy[1] + obj2_xy[1]) / 2,
        0.8  # 假设物体在桌面上方（高度约0.8m）
    ])
    
    # 计算物体之间的距离
    obj_distance = np.linalg.norm(np.array(obj1_xy) - np.array(obj2_xy))
    
    # 机械臂在世界坐标系中的位置
    robot_world = np.array([robot_xy[0], robot_xy[1], 0.0])
    
    # 当前相机在世界坐标系中的位置（近似）
    current_camera_world = robot_world + default_camera_pos
    
    # 计算需要的视野宽度（要看到两个物体 + 一些边界）
    required_view_width = obj_distance * 1.5  # 1.5倍余量
    
    # 根据相机到物体中心的距离，计算需要的FOV
    camera_to_obj_distance = np.linalg.norm(current_camera_world[:2] - obj_center[:2])
    
    # FOV计算：tan(FOV/2) = (view_width/2) / distance
    required_fov = 2 * np.arctan(required_view_width / (2 * camera_to_obj_distance))
    default_fov = np.radians(60)  # 假设默认FOV为60度
    fov_multiplier = required_fov / default_fov
    
    print(f"\n📊 相机视野分析:")
    print(f"   物体1位置: [{obj1_xy[0]:.3f}, {obj1_xy[1]:.3f}]")
    print(f"   物体2位置: [{obj2_xy[0]:.3f}, {obj2_xy[1]:.3f}]")
    print(f"   物体间距: {obj_distance:.3f}m")
    print(f"   需要视野宽度: {required_view_width:.3f}m")
    print(f"   相机到物体距离: {camera_to_obj_distance:.3f}m")
    print(f"   当前FOV: {np.degrees(default_fov):.1f}°")
    print(f"   需要FOV: {np.degrees(required_fov):.1f}°")
    print(f"   FOV放大倍数: {fov_multiplier:.2f}x")
    
    # 策略1: 如果物体距离太远，相机需要后退
    position_offset = [0.0, 0.0, 0.0]
    if fov_multiplier > 1.3:  # 需要放大FOV超过30%
        # 相机后退以扩大视野（沿y轴负方向）
        retreat_distance = 0.05  # 后退5cm
        position_offset[1] = -retreat_distance
        position_offset[2] = 0.03  # 同时升高3cm以保持俯视角
        print(f"   📹 策略: 相机后退 {retreat_distance*100:.0f}cm + 升高 3cm")
    
    # 策略2: 调整相机朝向，使其瞄准两个物体的中心
    # 计算从相机指向物体中心的方向向量
    camera_to_center = obj_center - (current_camera_world + position_offset)
    
    # 计算需要的俯仰角和偏航角
    # pitch (俯仰): 向下看的角度
    distance_horizontal = np.linalg.norm(camera_to_center[:2])
    pitch_angle = -np.arctan2(camera_to_center[2], distance_horizontal)  # 负号表示向下
    
    # yaw (偏航): 左右转向
    yaw_angle = np.arctan2(camera_to_center[1], camera_to_center[0])
    
    rotation_offset = [0.0, pitch_angle, yaw_angle]
    
    print(f"   📐 朝向调整: pitch={np.degrees(pitch_angle):.1f}°, yaw={np.degrees(yaw_angle):.1f}°")
    
    return {
        'position_offset': position_offset,
        'rotation_offset': rotation_offset,
        'fov_multiplier': min(fov_multiplier, 1.5),  # 限制最大1.5倍
        'analysis': {
            'obj_distance': obj_distance,
            'camera_to_obj_distance': camera_to_obj_distance,
            'required_fov_degrees': np.degrees(required_fov)
        }
    }


def apply_camera_optimization(options, camera_params):
    """
    将相机优化参数应用到options配置中
    
    注意：ManiSkill2的相机参数是在机器人配置类中定义的，
    不能通过options直接修改。这个函数生成建议参数供手动修改代码使用。
    """
    print(f"\n🔧 相机优化参数（需要修改代码应用）:")
    print(f"   位置偏移: {camera_params['position_offset']}")
    print(f"   旋转偏移: [roll={np.degrees(camera_params['rotation_offset'][0]):.1f}°, "
          f"pitch={np.degrees(camera_params['rotation_offset'][1]):.1f}°, "
          f"yaw={np.degrees(camera_params['rotation_offset'][2]):.1f}°]")
    print(f"   FOV倍数: {camera_params['fov_multiplier']:.2f}x")
    
    # 计算新的相机位置
    default_pos = [0.00, -0.16, 0.336]
    new_pos = [
        default_pos[0] + camera_params['position_offset'][0],
        default_pos[1] + camera_params['position_offset'][1],
        default_pos[2] + camera_params['position_offset'][2]
    ]
    
    print(f"\n📝 修改建议:")
    print(f"   在 ManiSkill2_real2sim/mani_skill2_real2sim/agents/configs/widowx/defaults.py")
    print(f"   WidowXBridgeDatasetCameraSetupConfig.cameras 中修改:")
    print(f"   ")
    print(f"   原始:")
    print(f"   p=[0.00, -0.16, 0.336],")
    print(f"   ")
    print(f"   修改为:")
    print(f"   p=[{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}],")
    
    # 如果需要调整FOV，需要修改intrinsic矩阵
    if camera_params['fov_multiplier'] > 1.1:
        print(f"\n   ⚠️  需要放大FOV，建议修改intrinsic参数:")
        print(f"   focal_length 需要除以 {camera_params['fov_multiplier']:.2f}")
        print(f"   ")
        print(f"   原始:")
        print(f"   intrinsic=np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])")
        print(f"   ")
        new_focal = 623.588 / camera_params['fov_multiplier']
        print(f"   修改为:")
        print(f"   intrinsic=np.array([[{new_focal:.3f}, 0, 319.501], [0, {new_focal:.3f}, 239.545], [0, 0, 1]])")
    
    return camera_params


def test_camera_view_with_params(episode_dir, camera_params):
    """测试应用相机参数后的效果"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path) as f:
        options = json.load(f)
    
    print(f"\n🧪 测试相机视角...")
    
    env = simpler_env.make('google_robot_move_near_customizable')
    obs, info = env.reset(seed=options.get('seed'), options=options)
    
    # 获取物体位置
    obj_positions = []
    for obj in env.unwrapped.episode_objs:
        obj_positions.append([obj.pose.p[0], obj.pose.p[1]])
        print(f"   📦 {obj.name}: [{obj.pose.p[0]:.3f}, {obj.pose.p[1]:.3f}, {obj.pose.p[2]:.3f}]")
    
    # 渲染当前视角
    img = get_image_from_maniskill2_obs_dict(env, obs)
    if img is not None:
        from PIL import Image
        output_path = os.path.join(episode_dir, "camera_view_analysis.jpg")
        Image.fromarray(img).save(output_path)
        print(f"   ✅ 当前视角已保存: {output_path}")
    
    env.close()
    
    # 分析物体是否在视野内（简单启发式）
    if len(obj_positions) >= 2:
        obj_distance = np.linalg.norm(np.array(obj_positions[0]) - np.array(obj_positions[1]))
        print(f"\n   物体间距: {obj_distance:.3f}m")
        if obj_distance > 0.3:
            print(f"   ⚠️  物体距离较远，可能需要调整相机")
        else:
            print(f"   ✅ 物体距离适中")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="优化相机视角")
    parser.add_argument("--episode_dir", type=str, required=True,
                       help="Episode目录路径")
    parser.add_argument("--test", action="store_true",
                       help="测试当前视角并生成优化建议")
    
    args = parser.parse_args()
    
    # 加载episode配置
    options_path = os.path.join(args.episode_dir, "options.json")
    if not os.path.exists(options_path):
        print(f"❌ 错误: 找不到 {options_path}")
        return
    
    with open(options_path) as f:
        options = json.load(f)
    
    # 获取机械臂和物体位置
    robot_xy = options.get("robot_init_options", {}).get("init_xy", [0.0, 0.0])
    
    # 从环境中获取物体位置
    print(f"🔍 分析场景配置...")
    print(f"   机械臂位置: [{robot_xy[0]:.3f}, {robot_xy[1]:.3f}]")
    
    env = simpler_env.make('google_robot_move_near_customizable')
    obs, info = env.reset(seed=options.get('seed'), options=options)
    
    obj_positions = []
    for obj in env.unwrapped.episode_objs:
        obj_xy = [obj.pose.p[0], obj.pose.p[1]]
        obj_positions.append(obj_xy)
        print(f"   📦 {obj.name}: [{obj_xy[0]:.3f}, {obj_xy[1]:.3f}]")
    
    env.close()
    
    if len(obj_positions) < 2:
        print(f"\n⚠️  场景中物体数量不足2个，无需优化")
        return
    
    # 计算相机优化参数
    camera_params = calculate_camera_params_for_two_objects(
        robot_xy, obj_positions[0], obj_positions[1]
    )
    
    # 应用并显示优化建议
    apply_camera_optimization(options, camera_params)
    
    if args.test:
        test_camera_view_with_params(args.episode_dir, camera_params)
    
    print(f"\n✅ 分析完成")


if __name__ == "__main__":
    main()
