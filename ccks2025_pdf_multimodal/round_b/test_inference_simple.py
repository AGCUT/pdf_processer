#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的推理测试脚本
用于验证训练好的 Qwen3-VL 模型
"""

import os
import json
import numpy as np
import pandas as pd
import re
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["MAX_PIXELS"] = "602112"

# 模型路径 - 合并后的模型
MODEL_PATH = "/usr/yuque/guo/pdf_processer/qwen3_vl_32b_merged"

# 测试数据
TEST_DATA = "/usr/yuque/guo/pdf_processer/patent_b/test/test.jsonl"
BASE_DIR = "/usr/yuque/guo/pdf_processer/patent_b/test"

# 向量文件（用于相似度检索）
TEST_QUESTION_VECTORS = "/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/all_test_b_question_vectors.npy"
TEST_IMAGE_VECTORS = "/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/test_b_pdf_img_vectors.npy"
TEST_IMAGE_MAPPING = "/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/test_b_pdf_img_page_num_mapping.csv"

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

# 加载向量数据
print("加载向量数据...")
question_vector = np.load(TEST_QUESTION_VECTORS)
pdf_image_vectors = np.load(TEST_IMAGE_VECTORS)
pdf_image_page_num_mapping = pd.read_csv(TEST_IMAGE_MAPPING)
df_question = pd.read_json(TEST_DATA, lines=True)
print(f"测试问题数量: {len(df_question)}")
print("向量数据加载完成！\n")


def get_similar_image_embedding(document_name, question_idx, top_k=2, exclude_page=-1):
    """获取与问题最相似的图片页码"""
    doc_name = document_name.split('.')[0]
    vec_idx = pdf_image_page_num_mapping[pdf_image_page_num_mapping['file_name'] == doc_name]['index'].values

    if len(vec_idx) == 0:
        return []

    candidate_vec = pdf_image_vectors[vec_idx]
    query_vec = question_vector[question_idx]

    # 计算余弦相似度
    cos_sim = np.dot(candidate_vec, query_vec) / (
        np.linalg.norm(candidate_vec, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    # 获取最相似的索引
    top_k_indices = np.argsort(cos_sim)[-(top_k+1):][::-1]
    retrived_idx = vec_idx[top_k_indices]
    retrived_page_num = pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    retrived_page_num = [int(x) for x in retrived_page_num]

    # 排除指定页面
    if exclude_page >= 0 and exclude_page in retrived_page_num:
        retrived_page_num.remove(exclude_page)

    return sorted(retrived_page_num[:top_k])


def vllm_inference(messages, max_tokens=500):
    """使用 vLLM 进行推理"""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens
    )

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

    outputs = vl_model.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text


def get_image_answer(document_name, question, question_idx):
    """普通问题的推理（不含特定页码）"""
    prompt_prefix = "你是一个专利内容分析专家，请根据我提供的专利内容回答问题。\n"
    prompt_prefix += f"【问题】{question}\n"
    prompt_prefix += "【专利内容】\n"

    # 获取相似图片
    retrived_page_num = get_similar_image_embedding(document_name, question_idx, top_k=2)

    if len(retrived_page_num) == 0:
        return "无法找到相关图片"

    # 构建 messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_prefix}
        ]
    }]

    # 添加图片
    doc_name = document_name.split('.')[0]
    for page_num in retrived_page_num:
        img_path = os.path.join(BASE_DIR, 'pdf_img', doc_name, f'{page_num}.jpg')
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

    return vllm_inference(messages)


def get_mix_answer_img(document_name, pic_page_num, question, question_idx):
    """含特定页码的问题推理"""
    prompt_prefix = "你是一个专利内容分析专家，请根据我提供的专利内容回答问题。\n"
    prompt_prefix += f"【问题】{question}\n"
    prompt_prefix += "【该问题指向的专利页内容】\n"

    doc_name = document_name.split('.')[0]
    main_image = os.path.join(BASE_DIR, 'pdf_img', doc_name, f'{pic_page_num}.jpg')

    if not os.path.exists(main_image):
        return "无法找到指定页面的图片"

    # 构建 messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_prefix},
            {"type": "image", "image": main_image, "max_pixels": 602112}
        ]
    }]

    # 获取其他相关图片
    retrived_page_num = get_similar_image_embedding(document_name, question_idx, top_k=2, exclude_page=pic_page_num)

    if len(retrived_page_num) > 0:
        messages[0]['content'].append({"type": "text", "text": "\n【其他相关专利内容】\n"})
        for page_num in retrived_page_num:
            img_path = os.path.join(BASE_DIR, 'pdf_img', doc_name, f'{page_num}.jpg')
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

    return vllm_inference(messages)


# 测试前5个样本
print("=" * 60)
print("开始测试前5个样本...")
print("=" * 60)

results = []

for i in range(min(5, len(df_question))):
    print(f"\n{'='*60}")
    print(f"样本 {i}")
    print(f"{'='*60}")

    question = df_question.loc[i, 'question']
    document = df_question.loc[i, 'document']

    print(f"问题: {question}")
    print(f"文档: {document}")

    # 判断问题类型并推理
    if "第" in question and "页" in question:
        # 问题指向特定页面
        page_match = re.findall(r"第(\d+)页", question)
        if page_match:
            pic_page_num = int(page_match[0])
            print(f"检测到指向第 {pic_page_num} 页的问题")
            answer = get_mix_answer_img(document, pic_page_num, question, i)
        else:
            answer = get_image_answer(document, question, i)
    else:
        answer = get_image_answer(document, question, i)

    print(f"\n模型回答: {answer}")

    results.append({
        'idx': i,
        'document': document,
        'question': question,
        'answer': answer
    })

# 保存结果
output_file = 'test_inference_results.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"\n{'='*60}")
print(f"测试完成！结果已保存到: {output_file}")
print(f"{'='*60}")
