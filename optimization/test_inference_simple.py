#!/usr/bin/env python3
"""
简化版推理测试，用于诊断问题
"""

import os
import sys
import json
import time
from pathlib import Path

# 设置 headless 渲染器
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ.pop('DISPLAY', None)
os.environ.pop('WAYLAND_DISPLAY', None)

# 设置项目路径
project_root = Path("/mnt/disk1/decom/VLATest")
sys.path.insert(0, str(project_root))

def test_dataset_loading():
    """测试数据集加载"""
    dataset_file = project_root / "optimization/grasp/batch_grasp_dataset.json"
    
    print(f"📋 测试数据集: {dataset_file}")
    
    if not dataset_file.exists():
        print("❌ 数据集文件不存在")
        return False
    
    try:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ 数据集加载成功")
        print(f"   场景数量: {data.get('num', 0)}")
        print(f"   种子: {data.get('seed', '未设置')}")
        
        # 显示所有场景
        scene_keys = [k for k in data.keys() if k.isdigit()]
        print(f"   场景ID: {', '.join(scene_keys)}")
        
        # 检查第一个场景
        if scene_keys:
            first_scene = data[scene_keys[0]]
            print(f"\n🔍 第一个场景详情:")
            print(f"   model_ids: {first_scene.get('model_ids', [])}")
            print(f"   source_obj_id: {first_scene.get('source_obj_id', 'N/A')}")
            
            # 获取源物体
            model_ids = first_scene.get('model_ids', [])
            source_id = first_scene.get('source_obj_id', 0)
            if isinstance(model_ids, list) and 0 <= source_id < len(model_ids):
                source_obj = model_ids[source_id]
                print(f"   源物体: {source_obj}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simpler_env():
    """测试 simpler_env 是否能正常初始化"""
    print(f"\n🔧 测试 simpler_env...")
    
    try:
        # 延迟导入，避免环境变量设置太晚
        from simpler_env import make
        
        # 尝试创建一个简单的环境
        print("   创建 'google_robot_grasp_customizable' 环境...")
        
        start_time = time.time()
        env = make(
            "google_robot_grasp_customizable",
            obs_mode="rgbd",
            max_episode_steps=50
        )
        
        print(f"   ✅ 环境创建成功 ({time.time()-start_time:.1f}s)")
        print(f"   动作空间: {env.action_space}")
        print(f"   观测空间: {env.observation_space}")
        
        return env
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_episode():
    """测试单个 episode"""
    print(f"\n🚀 测试单个 episode 推理...")
    
    try:
        # 导入 openVLA 相关模块
        from experiments.model_interface import OpenVLAInterface
        
        # 创建接口
        print("   创建 OpenVLA 接口...")
        vla = OpenVLAInterface(
            model_name="openvla-7b",
            task="google_robot_grasp_customizable"
        )
        
        print("   ✅ 接口创建成功")
        print(f"   模型: {vla.model_name}")
        print(f"   任务: {vla.task}")
        
        # 尝试运行一个简单的推理
        print("\n   🔄 尝试运行推理...")
        
        # 读取数据集中的第一个场景
        dataset_file = project_root / "optimization/grasp/batch_grasp_dataset.json"
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        scene_keys = [k for k in data.keys() if k.isdigit()]
        if not scene_keys:
            print("   ❌ 数据集中没有场景")
            return False
        
        first_scene_id = scene_keys[0]
        options = data[first_scene_id]
        
        print(f"   使用场景 {first_scene_id}")
        print(f"   源物体: {options.get('model_ids', [])[options.get('source_obj_id', 0)]}")
        
        # 运行推理（只运行几步测试）
        images, episode_stats, actions = vla.run_interfaceWithPromot(
            seed=options.get('seed', 0),
            options=options,
            promot="Grasp the object."  # 简单的提示
        )
        
        print(f"   ✅ 推理成功！")
        print(f"   步数: {len(actions)}")
        print(f"   图像数: {len(images)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("🔍 推理诊断工具")
    print("=" * 70)
    
    # 1. 测试数据集
    if not test_dataset_loading():
        return 1
    
    # 2. 测试环境创建
    env = test_simpler_env()
    if env is None:
        print("\n⚠️  环境创建失败，尝试修复...")
        
        # 检查依赖
        print("\n📦 检查依赖:")
        try:
            import sapien
            print("   ✅ sapien: OK")
        except Exception as e:
            print(f"   ❌ sapien: {e}")
            
        try:
            import OpenGL
            print("   ✅ OpenGL: OK")
        except Exception as e:
            print(f"   ❌ OpenGL: {e}")
            
        try:
            import OpenGL.EGL
            print("   ✅ OpenGL.EGL: OK")
        except Exception as e:
            print(f"   ❌ OpenGL.EGL: {e}")
            
        return 1
    
    # 3. 测试重置环境
    print(f"\n🔄 测试环境重置...")
    try:
        start_time = time.time()
        obs, info = env.reset(seed=42)
        reset_time = time.time() - start_time
        
        print(f"   ✅ 环境重置成功 ({reset_time:.1f}s)")
        print(f"   观测键: {list(obs.keys())}")
        
        # 尝试一个随机动作
        print(f"\n🎮 测试随机动作...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✅ 动作执行成功")
        print(f"   奖励: {reward}")
        print(f"   终止: {terminated}")
        
        env.close()
        print(f"   🔒 环境已关闭")
        
    except Exception as e:
        print(f"❌ 环境操作失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. 如果上面都成功，测试推理
    print(f"\n{'='*70}")
    print("🎯 开始完整推理测试")
    print("=" * 70)
    
    success = test_single_episode()
    
    if success:
        print(f"\n{'='*70}")
        print("✅ 所有测试通过！")
        print("=" * 70)
        return 0
    else:
        print(f"\n{'='*70}")
        print("❌ 测试失败")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())