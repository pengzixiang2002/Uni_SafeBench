#!/bin/bash

# ================= 1. 基础路径配置 =================
INPUT_DIR="./result"
EVAL_DIR="./eval"
SUM_DIR="./sum"

# ================= 2. 核心执行函数 =================
# 参数: $1=相对路径 (例如 "VQA/unsafety_I_safety_T"), $2=文件名
run_safety_checker() {
    local sub_path=$1
    local file_name=$2
    
    local in_file="${INPUT_DIR}/${sub_path}/${file_name}"
    local out_file="${EVAL_DIR}/${sub_path}/${file_name}"
    local stat_file="${SUM_DIR}/${sub_path}/${file_name}"

    # 检查输入文件是否存在
    if [ ! -f "$in_file" ]; then
        echo -e "\033[31m[Skip] File not found: $in_file\033[0m"
        return
    fi

    # 自动创建输出目录
    mkdir -p "$(dirname "$out_file")"
    mkdir -p "$(dirname "$stat_file")"

    echo "[Processing] ${sub_path}/${file_name} ..."
    
    python safety_checker.py \
        --input "$in_file" \
        --output "$out_file" \
        --stats_output "$stat_file"
}

# ================= 3. 各项任务处理 =================

# --- 任务 A: Text Generation ---
echo ">>> Starting Text Generation tasks..."
TEXT_GEN_FILES=(
    "uni_bagel.jsonl" "uni_chameleon.jsonl" "uni_deepseek-vl-base.jsonl" "uni_deepseek-vl-chat.jsonl" "uni_emu3.jsonl" "uni_gpt4o.jsonl" 
    "uni_harmon.jsonl" "uni_janus.jsonl" "uni_janusflow.jsonl" "uni_qwenvl.jsonl" "uni_showo2.jsonl" "uni_tar1.5b.jsonl" "uni_tar7b.jsonl" 
    "uni-deepseek-llm-chat.jsonl" "uni_qwen.jsonl" "uni-gemini.jsonl"
)
for file in "${TEXT_GEN_FILES[@]}"; do
    run_safety_checker "Text Generation" "$file"
done


# --- 任务 B: T2I (Text to Image) ---
# 注意：T2I 的路径包含模型子文件夹，直接调用函数指定路径
echo -e "\n>>> Starting T2I tasks..."
run_safety_checker "T2I/Bagel" "uni_bagel.jsonl"
run_safety_checker "T2I/Emu3" "uni_emu3.jsonl"
run_safety_checker "T2I/Gemini" "uni_gemini.jsonl"
run_safety_checker "T2I/Harmon" "uni_harmon.jsonl"
run_safety_checker "T2I/Janus-Pro" "uni_janus.jsonl"
run_safety_checker "T2I/JanusFlow" "uni_janusflow.jsonl"
run_safety_checker "T2I/Show-o" "uni_showo.jsonl"
run_safety_checker "T2I/Tar7B" "uni_tar7b.jsonl"
run_safety_checker "T2I/Tar1.5B" "uni_tar1.5b.jsonl"

#  --- 任务 C: VQA (多场景嵌套) ---
echo -e "\n>>> Starting VQA tasks..."
VQA_SCENARIOS=(
    "safety_I_safety_T"
    "unsafety_I_safety_T"
    "unsafety_I_unsafety_T"
    "safety_I_unsafety_T"
)
VQA_FILES=(
    "uni_deepseek-vl-base.jsonl" 
    "uni_deepseek-vl-chat.jsonl" 
    "uni_janus.jsonl" 
    "uni_tar7b.jsonl"
    "uni_bagel.jsonl"
    "uni_qwenvl.jsonl"
    "uni_tar1.5b.jsonl"
    "uni_chameleon.jsonl"
    "uni_emu3.jsonl"
    "uni_gemini.jsonl"
    "uni_gpt4o.jsonl"
    "uni_harmon.jsonl"
    "uni_janusflow.jsonl"
    "uni_showo.jsonl"
)

for scenario in "${VQA_SCENARIOS[@]}"; do
    echo "--- Scenario: $scenario ---"
    for file in "${VQA_FILES[@]}"; do
        run_safety_checker "VQA/${scenario}" "$file"
    done
done

# --- 任务 D: Text-Guided Image Editing ---
echo -e "\n>>> Starting Text-Guided Image Editing tasks..."
run_safety_checker "Text-Guided Image Editing/Bagel" "uni_bagel.jsonl"
run_safety_checker "Text-Guided Image Editing/Tar7B" "uni_tar7b.jsonl"
run_safety_checker "Text-Guided Image Editing/Gemini" "uni_gemini.jsonl"

echo -e "\n========================================"
echo "All tasks completed!"
echo "========================================"