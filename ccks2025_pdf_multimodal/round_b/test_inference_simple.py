#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的推理测试脚本
用于验证训练好的 Qwen3-VL 模型
"""

import os
import json
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["MAX_PIXELS"] = "602112"

# 模型路径 - 需要指定合并后的模型
MODEL_PATH = "/usr/yuque/guo/pdf_processer/lora_output_qwen3_vl/v3-20251201-032211/checkpoint-45-merged"
# 测试数据
TEST_DATA = "/usr/yuque/guo/pdf_processer/patent_b/test/test.jsonl"
BASE_DIR = "/usr/yuque/guo/pdf_processer/patent_b/test"

print("=" * 60)
print("加载模型...")
print(f"模型路径: {MODEL_PATH}")
print("=" * 60)

# 加载模型
vl_model = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 3},
    gpu_memory_utilization=0.9,
    tensor_parallel_size=4,
    max_model_len=8192,
    max_num_seqs=1
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("模型加载完成！\n")


def test_single_sample(question_text, image_paths):
    """测试单个样本"""

    # 构建 prompt
    prompt_text = f"""你是一个专利内容分析专家，请根据我提供的专利内容回答问题。
【问题】{question_text}
【专利内容】
"""

    # 构建 messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text}
        ]
    }]

    # 添加图片
    for img_path in image_paths:
        if os.path.exists(img_path):
            messages[0]['content'].append({
                "type": "image",
                "image": img_path,
                "max_pixels": 602112
            })

    messages[0]['content'].append({
        "type": "text",
        "text": "\n\n请根据以上专利内容，简洁准确地回答问题："
    })

    # 处理输入
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    # 推理
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=500
    )

    outputs = vl_model.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    return generated_text


# 测试前3个样本
print("=" * 60)
print("开始测试前3个样本...")
print("=" * 60)

df_test = pd.read_json(TEST_DATA, lines=True)

for i in range(min(3, len(df_test))):
    print(f"\n{'='*60}")
    print(f"样本 {i}")
    print(f"{'='*60}")

    question = df_test.loc[i, 'question']
    document = df_test.loc[i, 'document']

    print(f"问题: {question}")
    print(f"文档: {document}")

    # 简单示例：使用文档的前2页图片
    doc_name = document.split('.')[0]
    image_paths = [
        os.path.join(BASE_DIR, 'pdf_img', doc_name, '1.jpg'),
        os.path.join(BASE_DIR, 'pdf_img', doc_name, '2.jpg')
    ]

    # 检查图片是否存在
    existing_images = [img for img in image_paths if os.path.exists(img)]

    if len(existing_images) == 0:
        print("警告: 未找到图片，跳过")
        continue

    print(f"使用图片: {len(existing_images)} 张")

    # 推理
    answer = test_single_sample(question, existing_images)

    print(f"\n模型回答: {answer}")
    print(f"{'='*60}\n")

print("\n测试完成！")
