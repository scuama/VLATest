# optimization/debug_inference.py
#!/usr/bin/env python3
"""
完整的推理调试工具
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

# 设置环境变量
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ.pop('DISPLAY', None)
os.environ.pop('WAYLAND_DISPLAY', None)

PROJECT_ROOT = Path("/mnt/disk1/decom/VLATest")
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试导入...")
    
    modules_to_test = [
        ('simpler_env', 'simpler_env'),
        ('experiments.model_interface', 'OpenVLAInterface'),
        ('ManiSkill2_real2sim', 'ManiSkill2_real2sim'),
        ('sapien', 'sapien'),
        ('OpenGL', 'OpenGL'),
        ('gymnasium', 'gymnasium'),
    ]
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except Exception as e:
            print(f"  ❌ {display_name}: {e}")
    
    return True

def test_simpler_env_direct():
    """直接测试 simpler_env.make"""
    print("\n🔧 直接测试 simpler_env.make...")
    
    try:
        import simpler_env
        
        print("  1. 创建 'google_robot_grasp_customizable' 环境...")
        start = time.time()
        env = simpler_env.make("google_robot_grasp_customizable")
        elapsed = time.time() - start
        
        print(f"  ✅ 创建成功 ({elapsed:.1f}s)")
        print(f"     类型: {type(env)}")
        print(f"     动作空间: {env.action_space}")
        
        print("\n  2. 测试环境重置...")
        start = time.time()
        obs, info = env.reset(seed=42)
        elapsed = time.time() - start
        
        print(f"  ✅ 重置成功 ({elapsed:.1f}s)")
        print(f"     观测键: {list(obs.keys())}")
        
        print("\n  3. 测试单步动作...")
        start = time.time()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        elapsed = time.time() - start
        
        print(f"  ✅ 动作执行成功 ({elapsed:.1f}s)")
        print(f"     奖励: {reward}")
        print(f"     终止: {terminated}")
        
        env.close()
        print("  🔒 环境已关闭")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_openvla_interface():
    """测试 OpenVLA 接口"""
    print("\n🤖 测试 OpenVLAInterface...")
    
    try:
        from experiments.model_interface import OpenVLAInterface
        
        print("  1. 创建接口...")
        vla = OpenVLAInterface(
            model_name="openvla-7b",
            task="google_robot_grasp_customizable"
        )
        
        print(f"  ✅ 接口创建成功")
        print(f"     任务: {vla.task}")
        print(f"     模型: {vla.model_name}")
        
        # 读取测试数据
        dataset_file = PROJECT_ROOT / "optimization/grasp/batch_grasp_dataset.json"
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        options = data['7']  # 使用 Episode 7
        
        print(f"\n  2. 准备测试数据...")
        print(f"     源物体: {options['model_ids'][options['source_obj_id']]}")
        print(f"     种子: {options.get('seed', 0)}")
        
        print(f"\n  3. 运行推理（限制3步）...")
        # 修改 max_episode_steps 为3以快速测试
        if 'max_episode_steps' in options:
            original_steps = options['max_episode_steps']
            options['max_episode_steps'] = 3
            print(f"     临时设置 max_episode_steps = 3（原为 {original_steps}）")
        
        start = time.time()
        try:
            images, episode_stats, actions = vla.run_interfaceWithPromot(
                seed=options.get('seed', 0),
                options=options,
                promot="Grasp the redbull can."
            )
            elapsed = time.time() - start
            
            print(f"  ✅ 推理成功！")
            print(f"     耗时: {elapsed:.1f}s")
            print(f"     步数: {len(actions)}")
            print(f"     图像: {len(images)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 接口测试失败: {e}")
        traceback.print_exc()
        return False

def test_full_openvla_script():
    """测试完整的 openVLA.py 脚本"""
    print("\n🚀 测试完整 openVLA.py 脚本...")
    
    try:
        # 创建测试命令
        dataset_file = PROJECT_ROOT / "optimization/grasp/batch_grasp_dataset.json"
        output_dir = PROJECT_ROOT / "optimization/grasp/debug_output"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "experiments/openVLA.py",
            "--data", str(dataset_file),
            "--output", str(output_dir),
            "--model", "openvla-7b"
        ]
        
        print(f"  命令: {' '.join(cmd)}")
        
        # 运行子进程
        import subprocess
        
        print(f"  启动进程...")
        start = time.time()
        
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ
        )
        
        # 读取输出
        line_count = 0
        timeout = 300  # 5分钟超时
        last_output_time = time.time()
        
        for line in process.stdout:
            line = line.rstrip()
            line_count += 1
            
            # 显示关键信息
            if any(keyword in line.lower() for keyword in 
                   ['error', 'exception', 'traceback', 'start', 'go', 'step', 'success']):
                print(f"      [{line_count}] {line[:100]}")
            
            last_output_time = time.time()
            
            # 检查超时
            if time.time() - start > timeout:
                print(f"  ⏱️  超时（{timeout}秒），终止进程...")
                process.terminate()
                break
        
        # 等待进程结束
        return_code = process.wait()
        elapsed = time.time() - start
        
        print(f"\n  进程结束:")
        print(f"     返回码: {return_code}")
        print(f"     运行时间: {elapsed:.1f}s")
        print(f"     输出行数: {line_count}")
        
        if return_code == 0:
            print(f"  ✅ 脚本执行成功")
            return True
        else:
            print(f"  ❌ 脚本执行失败")
            return False
        
    except Exception as e:
        print(f"❌ 脚本测试失败: {e}")
        traceback.print_exc()
        return False

def check_log_file():
    """检查日志文件"""
    print("\n📄 检查最新日志...")
    
    log_dir = PROJECT_ROOT / "optimization/grasp/log"
    if log_dir.exists():
        log_files = list(log_dir.glob("batch_inference_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"  最新日志: {latest_log.name}")
            
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            print(f"  总行数: {len(lines)}")
            
            if len(lines) > 0:
                print(f"\n  最后20行:")
                for line in lines[-20:]:
                    print(f"      {line.rstrip()}")
            
            # 查找错误
            errors = [i for i, line in enumerate(lines) if 'error' in line.lower() or 'exception' in line.lower()]
            if errors:
                print(f"\n  发现 {len(errors)} 个错误:")
                for i in errors[:5]:  # 显示前5个错误
                    print(f"      第{i+1}行: {lines[i].strip()}")
    
    return True

def main():
    print("=" * 70)
    print("🔍 完整的推理调试工具")
    print("=" * 70)
    
    # 1. 测试导入
    test_imports()
    
    # 2. 检查日志
    check_log_file()
    
    # 3. 直接测试 simpler_env
    print("\n" + "=" * 70)
    print("阶段1: 测试 simpler_env")
    print("=" * 70)
    
    if not test_simpler_env_direct():
        print("\n❌ simpler_env 测试失败，无法继续")
        return 1
    
    # 4. 测试 OpenVLA 接口
    print("\n" + "=" * 70)
    print("阶段2: 测试 OpenVLA 接口")
    print("=" * 70)
    
    if not test_openvla_interface():
        print("\n❌ OpenVLA 接口测试失败")
        # 继续尝试完整脚本测试
    
    # 5. 测试完整脚本
    print("\n" + "=" * 70)
    print("阶段3: 测试完整脚本")
    print("=" * 70)
    
    test_full_openvla_script()
    
    print("\n" + "=" * 70)
    print("📋 调试完成")
    print("=" * 70)
    
    print("\n🔧 建议:")
    print("1. 如果 simpler_env 测试成功，说明基础环境正常")
    print("2. 如果 OpenVLA 接口失败，可能是模型加载问题")
    print("3. 检查 GPU 内存是否足够")
    print("4. 检查 HuggingFace 模型下载是否完整")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())