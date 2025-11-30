#!/bin/bash
# Qwen3-VL 模型微调脚本（多卡分布式训练）
# 使用 torchrun 进行真正的多卡并行训练
# 适配 4 × A800 80GB

# ============================================
# 配置路径
# ============================================

MODEL_PATH="/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen3-VL-32B-Instruct"
DATASET_PATH="/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/train_dataset_for_image.jsonl"
OUTPUT_DIR="/usr/yuque/guo/pdf_processer/lora_output_qwen3_vl"

# GPU配置 - 使用4张卡
GPUS="4,5,6,7"
NPROC_PER_NODE=4  # 4张GPU

echo "=============================================="
echo "开始模型微调 (Qwen3-VL - 分布式训练)"
echo "=============================================="
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用GPU: ${GPUS}"
echo "GPU数量: ${NPROC_PER_NODE}"
echo "日志文件: ${OUTPUT_DIR}/training.log"
echo "=============================================="

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 设置环境变量
export MAX_PIXELS=1229312
export CUDA_VISIBLE_DEVICES=${GPUS}

# 使用 torchrun 启动分布式训练
nohup torchrun \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_port 29500 \
    $(which swift) sft \
    --model ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --train_type lora \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.1 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 3 \
    --logging_steps 10 \
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
    --ddp_backend nccl \
    > ${OUTPUT_DIR}/training.log 2>&1 &

TRAIN_PID=$!
echo "分布式训练已在后台启动，进程PID: ${TRAIN_PID}"
echo "查看实时日志: tail -f ${OUTPUT_DIR}/training.log"
echo "查看进程状态: ps -p ${TRAIN_PID}"
echo "查看GPU使用: watch -n 1 nvidia-smi"
echo "=============================================="