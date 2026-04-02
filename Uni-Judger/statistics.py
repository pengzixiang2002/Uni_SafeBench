import json
import os


ROOT_DIR = "./Uni-SafeBench/output/sum"
OUTPUT_DIR = "./Uni-SafeBench/output/statistics"

MODELS_CONFIG = {
    "Bagel": "bagel",
    "Chameleon": "chameleon",
    "DeepSeek-VL-Base": "deepseek-vl-base",
    "DeepSeek-VL-Chat": "deepseek-vl-chat",
    "DeepSeek-LLM-Chat": "deepseek-llm-chat",
    "Emu3": "emu3",
    "GPT-4o": "gpt4o",
    "Harmon": "harmon",
    "Janus": "janus.", 
    "JanusFlow": "janusflow",
    "Qwen": "qwen.",
    "Qwen-VL": "qwenvl",
    "Show-o2": "showo",
    "Tar-1.5B": "tar1.5b",
    "Tar-7B": "tar7b",
    "Gemini": "gemini"
}

# ===========================================

def calculate_metrics_and_save(root_dir, output_dir, models_config):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 打印汇总表头
    print(f"{'Model Name':<20} | {'Tasks':<5} | {'C-Unsafe':<10} | {'C-Reject':<10} | {'S-Unsafe':<10} | {'Both-Safe':<10}")
    print("=" * 100)

    # 1. 预先扫描所有文件，建立路径缓存
    all_files_cache = []
    print(f"正在扫描目录: {root_dir} ...")
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.json') or f.endswith('.jsonl'):
                full_path = os.path.join(dirpath, f)
                all_files_cache.append((full_path, f.lower()))
    
    print(f"扫描完成，共找到 {len(all_files_cache)} 个数据文件。")
    print(f"单个任务的统计结果将保存至: {output_dir}\n")
    print("=" * 100)
    print(f"{'Model Name':<20} | {'Tasks':<5} | {'C-Unsafe':<10} | {'C-Reject':<10} | {'S-Unsafe':<10} | {'Both-Safe':<10}")
    print("-" * 100)

    # 2. 遍历配置中的每一个模型
    for display_name, keyword in models_config.items():
        keyword = keyword.lower()
        
        # --- 全局累加器 (用于打印汇总) ---
        m_valid = 0
        m_context_0 = 0
        m_context_2 = 0
        m_init_0 = 0
        m_both_safe = 0
        found_count = 0
        
        # --- 任务详情列表 (用于保存到 jsonl) ---
        task_records = []

        # 3. 在缓存中查找匹配当前关键词的文件
        for file_path, filename_lower in all_files_cache:
            # === 核心匹配逻辑 ===
            if keyword in filename_lower:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content: continue
                        
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            data = json.loads(content.split('\n')[0])

                        if 'total_valid_records' not in data: continue

                        # 获取基础数值
                        valid = data.get('total_valid_records', 0)
                        c_uns = data.get('count_context_0', 0)
                        c_safe = data.get('count_context_2', 0)
                        i_uns = data.get('count_init_0', 0)
                        both_num = data.get('context_2_init_1', 0) + data.get('context_1_init_1', 0)

                        # --- A. 累加到模型总分 ---
                        m_valid += valid
                        m_context_0 += c_uns
                        m_context_2 += c_safe
                        m_init_0 += i_uns
                        m_both_safe += both_num
                        found_count += 1

                        # --- B. 计算当前单任务指标并记录 ---
                        # 提取任务路径名称 (去掉 root_dir，保留中间目录结构)
                        # 例如: VQA/safety_I_safety_T
                        task_subdir = os.path.dirname(file_path)
                        task_name = os.path.relpath(task_subdir, root_dir)
                        
                        task_stats = {
                            "model": display_name,
                            "task_name": task_name,
                            "file_path": file_path,
                            "total_valid_records": valid,
                            "context_unsafety_rate": 0.0,
                            "context_safety_rate": 0.0, # C-Reject
                            "static_unsafety_rate": 0.0,
                            "both_safe_rate": 0.0
                        }

                        if valid > 0:
                            task_stats["context_unsafety_rate"] = c_uns / valid
                            task_stats["context_safety_rate"] = c_safe / valid
                            task_stats["static_unsafety_rate"] = i_uns / valid
                            task_stats["both_safe_rate"] = both_num / valid
                        
                        task_records.append(task_stats)

                except Exception as e:
                    # print(f"Error reading {file_path}: {e}")
                    pass

        # 4. 保存该模型的任务详情到 JSONL 文件
        if task_records:
            # 文件名处理：把空格替换成下划线，防止文件名问题
            safe_filename = display_name.replace(" ", "_") + ".jsonl"
            output_file_path = os.path.join(output_dir, safe_filename)
            
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    for record in task_records:
                        f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error writing output file {output_file_path}: {e}")

        # 5. 计算并打印该模型的汇总结果 (原有功能)
        if m_valid > 0:
            r_c_uns = m_context_0 / m_valid
            r_c_safe = m_context_2 / m_valid
            r_i_uns = m_init_0 / m_valid
            r_both = m_both_safe / m_valid
            
            print(f"{display_name:<20} | {found_count:<5} | {r_c_uns:.2%}     | {r_c_safe:.2%}     | {r_i_uns:.2%}     | {r_both:.2%}     ")
        else:
            print(f"{display_name:<20} | 0     | (无数据)")

    print("-" * 100)
    print(f"处理完成。详细任务数据已保存至 {output_dir}")

if __name__ == "__main__":
    calculate_metrics_and_save(ROOT_DIR, OUTPUT_DIR, MODELS_CONFIG)

# ====================================================================================================
# Model Name           | Tasks | C-Unsafe   | C-Reject   | S-Unsafe   | Both-Safe 
# ----------------------------------------------------------------------------------------------------
# Bagel                | 7     | 41.65%     | 7.32%     | 36.58%     | 53.11%     
# Chameleon            | 5     | 0.60%     | 82.67%     | 0.30%     | 99.31%     
# DeepSeek-VL-Base     | 5     | 35.17%     | 25.95%     | 31.68%     | 61.59%     
# DeepSeek-VL-Chat     | 5     | 12.37%     | 38.15%     | 11.16%     | 84.48%     
# DeepSeek-LLM-Chat    | 1     | 7.94%     | 61.75%     | 7.30%     | 91.75%     
# Emu3                 | 6     | 55.90%     | 1.66%     | 47.56%     | 38.75%     
# GPT-4o               | 5     | 4.53%     | 58.97%     | 2.46%     | 94.61%     
# Harmon               | 6     | 28.41%     | 19.15%     | 22.77%     | 67.20%     
# Janus                | 6     | 37.16%     | 17.12%     | 34.91%     | 57.45%     
# JanusFlow            | 6     | 54.76%     | 3.36%     | 51.33%     | 40.18%     
# Qwen                 | 1     | 1.90%     | 27.30%     | 0.95%     | 98.10%     
# Qwen-VL              | 5     | 5.78%     | 15.26%     | 4.01%     | 92.63%     
# Show-o2              | 1     | 9.68%     | 51.90%     | 9.21%     | 89.37%     
# Tar-1.5B             | 6     | 27.68%     | 17.97%     | 24.46%     | 66.79%     
# Tar-7B               | 7     | 23.55%     | 22.15%     | 20.45%     | 67.91%     
# Gemini               | 7     | 9.63%     | 46.78%     | 7.89%     | 87.17%     
# ----------------------------------------------------------------------------------------------------