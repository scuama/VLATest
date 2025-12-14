#!/usr/bin/env python3
"""
分析 move 任务结果并生成表格
"""
import json
import os
from pathlib import Path
import pandas as pd

# 配置
RESULTS_DIR = "results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024"
TASKS_JSON = "data/t-move_n-100_o-0_s-3225323079.json"
OUTPUT_CSV = "move_results_analysis.csv"
OUTPUT_SUMMARY = "move_results_summary.txt"

def load_tasks_info():
    """加载任务定义"""
    with open(TASKS_JSON, 'r') as f:
        data = json.load(f)
    
    # 转换为列表格式，提取物体名称
    tasks = []
    for task_id in range(100):
        task_data = data[str(task_id)]
        model_ids = task_data['model_ids']
        source_idx = task_data['source_obj_id']
        target_idx = task_data['target_obj_id']
        
        tasks.append({
            'source_obj_name': model_ids[source_idx],
            'target_obj_name': model_ids[target_idx],
            'task': f"move {model_ids[source_idx]} near {model_ids[target_idx]}"
        })
    
    return tasks

def analyze_single_task(task_id, task_dir):
    """分析单个任务的结果"""
    log_file = task_dir / "log.json"
    
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # log_data 是字典，键是步数(字符串)
        # 获取最后一步
        num_steps = len(log_data)
        last_step_key = str(num_steps - 1)
        last_step = log_data[last_step_key]
        
        # 辅助函数：将字符串"true"/"false"转为布尔值
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return bool(val)
        
        result = {
            'task_id': task_id,
            'success': to_bool(last_step.get('success', False)),
            'steps': num_steps,
            'all_obj_keep_height': to_bool(last_step.get('all_obj_keep_height', False)),
            'moved_correct_obj': to_bool(last_step.get('moved_correct_obj', False)),
            'near_tgt_obj': to_bool(last_step.get('near_tgt_obj', False)),
            'is_closest_to_tgt': to_bool(last_step.get('is_closest_to_tgt', False))
        }
        
        return result
    except Exception as e:
        print(f"⚠️  任务 {task_id} 分析出错: {e}")
        return None

def get_failure_reason(row):
    """确定失败原因"""
    if row['success']:
        return "成功"
    
    reasons = []
    if not row['all_obj_keep_height']:
        reasons.append("物体掉落/倾倒")
    if not row['moved_correct_obj']:
        reasons.append("未移动正确物体")
    if not row['near_tgt_obj']:
        reasons.append("未靠近目标")
    if not row['is_closest_to_tgt']:
        reasons.append("非最近物体")
    
    return " + ".join(reasons) if reasons else "未知原因"

def analyze_results():
    """分析所有任务结果"""
    tasks_info = load_tasks_info()
    results_list = []
    
    for task_id in range(100):
        task_dir = Path(RESULTS_DIR) / str(task_id)
        
        if not task_dir.exists():
            print(f"⚠️  任务 {task_id} 目录不存在")
            continue
        
        result = analyze_single_task(task_id, task_dir)
        if result:
            # 添加任务描述
            task_info = tasks_info[task_id]
            result['task_description'] = task_info['task']
            result['source_object'] = task_info['source_obj_name']
            result['target_object'] = task_info['target_obj_name']
            
            results_list.append(result)
    
    return results_list

def main():
    print("开始分析 move 任务结果...")
    
    # 分析结果
    results = analyze_results()
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 添加失败原因列
    df['failure_reason'] = df.apply(get_failure_reason, axis=1)
    
    # 重新排序列并重命名为中文
    df_chinese = df.copy()
    
    # 物体名称中英文映射表
    object_translation = {
        'bridge_spoon_generated_modified': '勺子',
        'bridge_carrot_generated_modified': '胡萝卜',
        'eggplant': '茄子',
        'opened_pepsi_can': '打开的百事可乐罐',
        'opened_coke_can': '打开的可口可乐罐',
        'opened_sprite_can': '打开的雪碧罐',
        'opened_fanta_can': '打开的芬达罐',
        'opened_7up_can': '打开的七喜罐',
        'opened_redbull_can': '打开的红牛罐',
        'pepsi_can': '百事可乐罐',
        'coke_can': '可口可乐罐',
        'sprite_can': '雪碧罐',
        'fanta_can': '芬达罐',
        'redbull_can': '红牛罐',
        'yellow_cube_3cm': '黄色方块(3cm)',
        'sponge': '海绵',
        'orange': '橙子',
        'blue_plastic_bottle': '蓝色塑料瓶',
    }
    
    # 将任务描述翻译为中文
    def translate_task_description(desc):
        # 先替换关键词
        desc = desc.replace('move', '移动')
        desc = desc.replace('near', '到')
        
        # 替换物体名称
        for eng, chn in object_translation.items():
            desc = desc.replace(eng, chn)
        
        # 清理下划线
        desc = desc.replace('_', ' ')
        return desc
    
    # 翻译任务描述
    df_chinese['task_description'] = df_chinese['task_description'].apply(translate_task_description)
    
    # 翻译物体名称（在重命名列之前）
    df_chinese['source_object'] = df_chinese['source_object'].map(lambda x: object_translation.get(x, x))
    df_chinese['target_object'] = df_chinese['target_object'].map(lambda x: object_translation.get(x, x))
    
    # 将布尔值转换为中文
    bool_columns = ['success', 'all_obj_keep_height', 'moved_correct_obj', 'near_tgt_obj', 'is_closest_to_tgt']
    for col in bool_columns:
        df_chinese[col] = df_chinese[col].map({True: '是', False: '否'})
    
    # 重命名列为中文
    df_chinese = df_chinese.rename(columns={
        'task_id': '任务ID',
        'task_description': '任务描述',
        'source_object': '源物体',
        'target_object': '目标物体',
        'success': '是否成功',
        'failure_reason': '失败原因',
        'steps': '执行步数',
        'all_obj_keep_height': '物体保持高度',
        'moved_correct_obj': '移动正确物体',
        'near_tgt_obj': '靠近目标',
        'is_closest_to_tgt': '是最近物体'
    })
    
    columns_order = [
        '任务ID', '任务描述', '源物体', '目标物体',
        '是否成功', '失败原因', '执行步数',
        '物体保持高度', '移动正确物体', '靠近目标', '是最近物体'
    ]
    df_chinese = df_chinese[columns_order]
    
    # 保存为 CSV
    df_chinese.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✓ 详细结果已保存到: {OUTPUT_CSV}")
    
    # 生成统计摘要
    total = len(df)
    success_count = df['success'].sum()
    fail_count = total - success_count
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    summary = f"""
{'='*80}
Move 任务结果分析摘要
{'='*80}

总体统计:
  - 总任务数: {total}
  - 成功: {success_count} ({success_rate:.1f}%)
  - 失败: {fail_count} ({100-success_rate:.1f}%)

失败原因统计:
"""
    
    # 统计失败原因
    failure_df = df[df['success'] == False]
    if len(failure_df) > 0:
        failure_reasons = failure_df['failure_reason'].value_counts()
        for reason, count in failure_reasons.items():
            pct = count / fail_count * 100
            summary += f"  - {reason}: {count} ({pct:.1f}%)\n"
        
        # 各个条件的失败统计
        summary += f"\n详细条件失败统计:\n"
        summary += f"  - 物体掉落/倾倒 (all_obj_keep_height=False): {(~df['all_obj_keep_height']).sum()}\n"
        summary += f"  - 未移动正确物体 (moved_correct_obj=False): {(~df['moved_correct_obj']).sum()}\n"
        summary += f"  - 未靠近目标 (near_tgt_obj=False): {(~df['near_tgt_obj']).sum()}\n"
        summary += f"  - 非最近物体 (is_closest_to_tgt=False): {(~df['is_closest_to_tgt']).sum()}\n"
    
    summary += f"\n平均步数: {df['steps'].mean():.1f}\n"
    summary += f"{'='*80}\n"
    
    print(summary)
    
    # 保存摘要
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✓ 摘要已保存到: {OUTPUT_SUMMARY}")
    
    # 显示前10个任务
    print("\n前10个任务详情:")
    display_df = df_chinese[['任务ID', '任务描述', '是否成功', '失败原因', '执行步数']].head(10)
    print(display_df.to_string(index=False))
    
    # 显示所有失败任务
    failure_df_chinese = df_chinese[df_chinese['是否成功'] == False]
    if len(failure_df_chinese) > 0:
        print(f"\n所有失败任务 ({len(failure_df_chinese)}个):")
        print(failure_df_chinese[['任务ID', '任务描述', '失败原因', '执行步数']].to_string(index=False))
    else:
        print("\n🎉 所有任务都成功了！")

if __name__ == "__main__":
    main()
