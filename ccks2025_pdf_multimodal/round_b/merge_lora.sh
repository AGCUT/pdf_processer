#!/bin/bash
# 合并 LoRA 权重到基础模型

# 配置路径
BASE_MODEL="/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen3-VL-32B-Instruct"
LORA_DIR="/usr/yuque/guo/pdf_processer/lora_output_qwen3_vl"
OUTPUT_DIR="/usr/yuque/guo/pdf_processer/qwen3_vl_merged"

echo "=============================================="
echo "合并 LoRA 权重"
echo "=============================================="
echo "基础模型: ${BASE_MODEL}"
echo "LoRA目录: ${LORA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

# 查找最新的 checkpoint
LATEST_CHECKPOINT=$(ls -td ${LORA_DIR}/v*/checkpoint-* 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到 checkpoint"
    exit 1
fi

echo "找到 checkpoint: ${LATEST_CHECKPOINT}"
echo ""

# 使用 swift 合并权重
swift export \
    --ckpt_dir ${LATEST_CHECKPOINT} \
    --merge_lora true

echo "=============================================="
echo "合并完成！"
echo "合并后的模型路径: ${LATEST_CHECKPOINT}-merged"
echo "=============================================="
