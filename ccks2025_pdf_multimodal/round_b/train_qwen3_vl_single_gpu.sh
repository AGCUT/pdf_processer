#!/bin/bash
# Qwen3-VL 模型微调脚本（单卡测试版）
# 用于排查分布式训练问题
# 使用 1 × A800 80GB

# ============================================
# 配置路径
# ============================================

MODEL_PATH="/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen3-VL-32B-Instruct"
DATASET_PATH="/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/train_dataset_for_image.jsonl"
OUTPUT_DIR="/usr/yuque/guo/pdf_processer/lora_output_qwen3_vl_single"

# GPU配置 - 只使用1张卡测试
GPUS="4"

echo "=============================================="
echo "开始模型微调 (Qwen3-VL - 单卡测试)"
echo "=============================================="
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用GPU: ${GPUS}"
echo "日志文件: ${OUTPUT_DIR}/training.log"
echo "=============================================="

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 设置环境变量
export MAX_PIXELS=1229312
export CUDA_VISIBLE_DEVICES=${GPUS}

# 单卡训练（不使用 torchrun）
nohup swift sft \
    --model ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --train_type lora \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.1 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --save_steps 20 \
    --eval_steps 20 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --seed 42 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-08 \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    > ${OUTPUT_DIR}/training.log 2>&1 &

TRAIN_PID=$!
echo "单卡训练已在后台启动，进程PID: ${TRAIN_PID}"
echo "查看实时日志: tail -f ${OUTPUT_DIR}/training.log"
echo "查看进程状态: ps -p ${TRAIN_PID}"
echo "查看GPU使用: watch -n 1 nvidia-smi"
echo "=============================================="
