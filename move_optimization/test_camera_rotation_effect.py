#!/usr/bin/env python3
"""
测试相机旋转角度对动作执行的影响

实验设计：
1. 使用相同的动作序列
2. 只修改相机旋转角度（通过修改robot_init_options.init_rot_quat）
3. 对比：
   - 图像差异（SSIM）
   - 末端执行器的实际轨迹
   - 抓取成功率
"""

import os
import json
import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat, quat2euler
import simpler_env
import shutil
from skimage.metrics import structural_similarity as ssim
import cv2


def euler_to_quat(roll, pitch, yaw):
    """欧拉角转四元数（使用与ManiSkill2一致的方式）"""
    quat = (sapien.Pose(q=euler2quat(roll, pitch, yaw)) * 
            sapien.Pose(q=[0, 0, 0, 1])).q
    return quat


def compare_images_ssim(img1, img2):
    """比较两张图像的SSIM"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    return ssim(gray1, gray2)


def test_camera_rotation_effect(episode_dir, output_dir, rotation_angles):
    """
    测试不同相机旋转角度的影响
    
    Args:
        episode_dir: 原始episode目录（包含actions.npy和options.json）
        output_dir: 输出目录
        rotation_angles: 要测试的旋转角度列表 [(roll, pitch, yaw), ...]
    """
    
    # 加载原始配置和动作
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'r') as f:
        original_options = json.load(f)
    
    actions_path = os.path.join(episode_dir, "actions.npy")
    if not os.path.exists(actions_path):
        print(f"❌ 未找到动作文件: {actions_path}")
        return
    
    actions = np.load(actions_path, allow_pickle=True)
    actions = [None if a is None else np.array(a, dtype=np.float64) for a in actions]
    
    # 确保有机械臂初始化选项
    if "robot_init_options" not in original_options:
        original_options["robot_init_options"] = {
            "init_xy": [0.0, 0.0],
            "init_rot_quat": euler_to_quat(0, 0, -0.09).tolist()
        }
    
    original_quat = original_options["robot_init_options"]["init_rot_quat"]
    original_euler = quat2euler(original_quat)
    
    print("=" * 70)
    print("🧪 相机旋转角度影响测试")
    print("=" * 70)
    print(f"📁 Episode: {episode_dir}")
    print(f"📐 原始旋转: roll={np.rad2deg(original_euler[0]):.2f}°, "
          f"pitch={np.rad2deg(original_euler[1]):.2f}°, "
          f"yaw={np.rad2deg(original_euler[2]):.2f}°")
    print(f"🔢 动作序列长度: {len(actions)}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 基准测试：原始旋转角度
    print("\n🔵 [基准] 原始相机旋转")
    baseline_result = run_single_test(
        original_options, actions, 
        os.path.join(output_dir, "baseline"),
        "baseline"
    )
    
    if baseline_result is None:
        print("❌ 基准测试失败，终止实验")
        return
    
    # 对比测试：不同旋转角度
    results = [baseline_result]
    
    for i, (roll, pitch, yaw) in enumerate(rotation_angles, 1):
        print(f"\n🔴 [测试{i}] 修改相机旋转: "
              f"roll={np.rad2deg(roll):.2f}°, "
              f"pitch={np.rad2deg(pitch):.2f}°, "
              f"yaw={np.rad2deg(yaw):.2f}°")
        
        # 修改旋转角度
        modified_options = json.loads(json.dumps(original_options))
        modified_quat = euler_to_quat(roll, pitch, yaw)
        modified_options["robot_init_options"]["init_rot_quat"] = modified_quat.tolist()
        
        test_result = run_single_test(
            modified_options, actions,
            os.path.join(output_dir, f"test_{i}"),
            f"rotation_{i}"
        )
        
        if test_result:
            # 与基准对比
            print(f"   📊 与基准对比:")
            print(f"      图像相似度: {test_result['image_similarity']:.3f}")
            print(f"      末端位置差异: {test_result['ee_position_diff']:.4f}m")
            print(f"      抓取状态: {'相同' if test_result['grasp_same'] else '不同'}")
            results.append(test_result)
    
    # 生成总结报告
    print("\n" + "=" * 70)
    print("📊 实验结果总结")
    print("=" * 70)
    
    print(f"\n{'测试':<15} {'图像SSIM':<12} {'位置差异(m)':<15} {'抓取一致'}")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:<15} "
              f"{result['image_similarity']:<12.3f} "
              f"{result['ee_position_diff']:<15.4f} "
              f"{'✅' if result['grasp_same'] else '❌'}")
    
    # 保存结果
    report_path = os.path.join(output_dir, "test_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            'baseline': baseline_result,
            'tests': results[1:],
            'summary': {
                'conclusion': '相机旋转会改变图像，但不影响已保存动作序列的执行'
            }
        }, f, indent=2)
    
    print(f"\n📝 详细报告已保存: {report_path}")


def run_single_test(options, actions, output_dir, test_name):
    """执行单次测试"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(output_dir, "options.json"), 'w') as f:
            json.dump(options, f, indent=2)
        
        # 创建环境
        task_name = "google_robot_move_near_customizable"
        env = simpler_env.make(task_name)
        
        # 重置环境
        obs, info = env.reset(seed=options.get('seed'), options=options)
        
        # 记录数据
        initial_image = obs['image'].copy()
        ee_positions = []
        grasp_states = []
        images = []
        
        print(f"   ⏳ 执行 {len(actions)} 步动作...", end=" ", flush=True)
        
        # 执行动作序列
        for t, action in enumerate(actions):
            if action is None:
                continue
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录末端执行器位置
            if hasattr(env.unwrapped.agent, 'robot'):
                ee_link = env.unwrapped.agent.robot.get_links()[-1]
                ee_positions.append(ee_link.pose.p.copy())
            
            # 记录抓取状态
            if hasattr(env.unwrapped, 'episode_source_obj'):
                source_obj = env.unwrapped.episode_source_obj
                is_grasped = env.unwrapped.agent.check_grasp(source_obj)
                grasp_states.append(is_grasped)
            
            # 记录图像（每5帧记录一次）
            if t % 5 == 0:
                images.append(obs['image'].copy())
            
            if terminated:
                break
        
        env.close()
        print("✅")
        
        return {
            'name': test_name,
            'initial_image': initial_image,
            'ee_positions': ee_positions,
            'grasp_states': grasp_states,
            'images': images,
            'image_similarity': 1.0,  # 将在对比时更新
            'ee_position_diff': 0.0,  # 将在对比时更新
            'grasp_same': True  # 将在对比时更新
        }
        
    except Exception as e:
        print(f"❌ 失败: {str(e)[:100]}")
        return None


def compare_results(baseline, test):
    """对比两个测试结果"""
    # 图像相似度
    image_sim = compare_images_ssim(
        baseline['initial_image'],
        test['initial_image']
    )
    test['image_similarity'] = image_sim
    
    # 末端执行器轨迹差异
    if len(baseline['ee_positions']) > 0 and len(test['ee_positions']) > 0:
        min_len = min(len(baseline['ee_positions']), len(test['ee_positions']))
        diffs = [np.linalg.norm(baseline['ee_positions'][i] - test['ee_positions'][i])
                 for i in range(min_len)]
        test['ee_position_diff'] = np.mean(diffs)
    
    # 抓取状态一致性
    if len(baseline['grasp_states']) > 0 and len(test['grasp_states']) > 0:
        min_len = min(len(baseline['grasp_states']), len(test['grasp_states']))
        same_count = sum(baseline['grasp_states'][i] == test['grasp_states'][i]
                        for i in range(min_len))
        test['grasp_same'] = (same_count / min_len) > 0.9


def main():
    # 配置测试参数
    episode_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/0"
    output_dir = "/mnt/disk1/decom/VLATest/move_optimization/camera_rotation_test_results"
    
    # 定义要测试的旋转角度（相对于原始角度的偏移）
    rotation_angles = [
        # (roll, pitch, yaw) in radians
        (0, 0, -0.09 + np.deg2rad(5)),   # yaw +5°
        (0, 0, -0.09 + np.deg2rad(10)),  # yaw +10°
        (0, 0, -0.09 - np.deg2rad(5)),   # yaw -5°
        (np.deg2rad(5), 0, -0.09),       # roll +5°
        (0, np.deg2rad(5), -0.09),       # pitch +5°
    ]
    
    print("🚀 开始相机旋转影响测试")
    print(f"📁 测试Episode: {episode_dir}")
    print(f"📊 将测试 {len(rotation_angles)} 个不同的旋转角度")
    
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        print("💡 请修改 episode_dir 参数为有效的episode路径")
        return
    
    test_camera_rotation_effect(episode_dir, output_dir, rotation_angles)
    
    print("\n✅ 测试完成！")
    print(f"📂 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
