#!/bin/bash
# 合并 LoRA 权重到基础模型

# 配置路径
BASE_MODEL="/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen3-VL-32B-Instruct"
CHECKPOINT_DIR="/usr/yuque/guo/pdf_processer/lora_output_qwen3_vl/v4-20251201-034347/checkpoint-45"
OUTPUT_DIR="/usr/yuque/guo/pdf_processer/qwen3_vl_32b_merged"

# GPU配置
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo "=============================================="
echo "合并 LoRA 权重到基础模型"
echo "=============================================="
echo "基础模型: ${BASE_MODEL}"
echo "LoRA Checkpoint: ${CHECKPOINT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

# 检查路径是否存在
if [ ! -d "${BASE_MODEL}" ]; then
    echo "错误: 基础模型目录不存在: ${BASE_MODEL}"
    exit 1
fi

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "错误: checkpoint 目录不存在: ${CHECKPOINT_DIR}"
    echo "请检查以下目录："
    ls -lh /usr/yuque/guo/pdf_processer/lora_output_qwen3_vl/
    exit 1
fi

echo "开始合并..."
echo ""

# 使用 swift export 合并 LoRA 权重
swift export \
    --model ${BASE_MODEL} \
    --adapters ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --merge_lora true

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "合并完成！"
    echo "合并后的模型路径: ${OUTPUT_DIR}"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "合并失败！请检查错误信息"
    echo "=============================================="
    exit 1
fi
