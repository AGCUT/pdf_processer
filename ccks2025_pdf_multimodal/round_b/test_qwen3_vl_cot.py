#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL 模型 Chain-of-Thought 推理测试脚本
用于测试 COT 对准确度的影响

COT策略：
1. 对位置关系问题使用分步推理prompt
2. 对部件识别问题使用分步推理prompt
3. 其他问题使用原始prompt（与test_qwen3_vl.py完全一致）

对照组（--no_cot）：与test_qwen3_vl.py完全一致的逻辑
"""

import numpy as np
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import re
import json
import argparse
from tqdm import trange

# ============================================
# 配置路径
# ============================================

# 训练集（用于答案风格匹配）
train_base_dir = '/usr/yuque/guo/pdf_processer/patent_b/train/'
df_train_question = pd.read_json("/usr/yuque/guo/pdf_processer/patent_b/train/train.jsonl", lines=True)
train_question_vector = np.load('all_train_b_question_vectors.npy')

# 测试集
base_dir = '/usr/yuque/guo/pdf_processer/patent_b/test/'
df_question = pd.read_json("/usr/yuque/guo/pdf_processer/patent_b/test/test.jsonl", lines=True)
question_vector = np.load('all_test_b_question_vectors.npy')

# 图片向量
test_pdf_image_vectors = np.load("test_b_pdf_img_vectors.npy")
test_pdf_image_page_num_mapping = pd.read_csv('test_b_pdf_img_page_num_mapping.csv')

# ============================================
# 模型配置
# ============================================

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["MAX_PIXELS"] = "602112"

# 合并后的模型路径
model_path = "/usr/yuque/guo/pdf_processer/qwen3_vl_32b_merged"

print("=" * 60)
print("加载 Qwen3-VL 模型...")
print(f"模型路径: {model_path}")
print("=" * 60)

vl_model = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 3},
    gpu_memory_utilization=0.9,
    tensor_parallel_size=4,
    max_model_len=8192,
    max_num_seqs=1
)
processor = AutoProcessor.from_pretrained(model_path)

print("模型加载完成！")
print(f"测试问题数量: {len(df_question)}")
print("=" * 60)


# ============================================
# 推理函数
# ============================================

def origin_vllm(messages, max_tokens=768):
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
    else:
        llm_inputs = {
            "prompt": prompt
        }

    outputs = vl_model.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text


# ============================================
# 相似度检索函数
# ============================================

def get_similar_question_embedding(question_idx, top_k=2):
    """获取与当前问题最相似的训练集问题索引"""
    query_vec = question_vector[question_idx]
    cos_sim = np.dot(train_question_vector, query_vec) / (
        np.linalg.norm(train_question_vector, axis=1) * np.linalg.norm(query_vec)
    )
    top_k_indices = np.argsort(cos_sim)[-(top_k):][::-1]
    return top_k_indices[:top_k]


def get_options_for_similar_answer(retrived_question_idx):
    """获取相似问题的答案作为风格参考"""
    options_str = '回答风格示例: '
    for idx in retrived_question_idx:
        ans = df_train_question.loc[idx, 'answer']
        options_str += (ans + '\n')
    return options_str + '\n\n'


def get_similar_image_embedding(document_name, question_idx, top_k, pic_page_num):
    """获取与问题最相似的图片页码"""
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = test_pdf_image_page_num_mapping[
        test_pdf_image_page_num_mapping['file_name'] == document_name
    ]['index'].values

    if len(vec_idx) == 0:
        return []

    candidate_vec = test_pdf_image_vectors[vec_idx]
    query_vec = question_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (
        np.linalg.norm(candidate_vec, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    top_k_indices = np.argsort(cos_sim)[-(top_k+1):][::-1]
    retrived_idx = vec_idx[top_k_indices]
    retrived_page_num = test_pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    retrived_page_num = [int(x) for x in retrived_page_num]

    if pic_page_num >= 0 and pic_page_num in retrived_page_num:
        retrived_page_num.remove(pic_page_num)

    retrived_page_num = retrived_page_num[:top_k]
    retrived_page_num = sorted(retrived_page_num)
    return retrived_page_num


# ============================================
# 问题类型分类
# ============================================

def classify_question_type(question):
    """分类问题类型"""
    position_keywords = ['位置', '方向', '上方', '下方', '左侧', '右侧', '哪里', '哪个位置',
                         '前方', '后方', '旁边', '之间', '内部', '外部', '顶部', '底部']
    identification_keywords = ['是什么', '什么部件', '哪个部件', '叫什么', '名称', '作用', '功能']

    if any(kw in question for kw in position_keywords):
        return 'position'
    elif any(kw in question for kw in identification_keywords):
        return 'identification'
    else:
        return 'other'


# ============================================
# COT Prompt 构建（仅用于COT模式）
# ============================================

def build_cot_prompt_position(question):
    """构建位置关系问题的COT提示词"""
    prompt = f"""你是一个专利分析专家。请仔细分析图片，一步步回答下面的位置关系问题。

问题：{question}

请按照以下步骤思考：

【步骤1：识别关键部件】
首先，我需要在图中找到问题提到的部件，并记下它们的编号位置。

【步骤2：观察空间位置】
然后，我需要仔细观察这些部件在图中的相对位置关系。注意：图中部件的上下、前后、左右位置判断应以标号线所指代的实际结构为准，而不是仅凭直观看数字。

【步骤3：确定方向关系】
接下来，我需要确定它们之间的方向关系（上下、左右、前后等）。

【步骤4：得出结论】
最后，基于以上观察，我可以给出准确的答案。

现在请开始你的分析，并在最后用【最终答案】标注你的结论（答案通常20字以内）："""
    return prompt


def build_cot_prompt_identification(question):
    """构建部件识别问题的COT提示词"""
    prompt = f"""你是一个专利分析专家。请仔细分析图片，一步步回答下面的部件识别问题。

问题：{question}

请按照以下步骤思考：

【步骤1：定位目标编号】
首先，我需要在图中找到问题提到的编号。

【步骤2：观察部件特征】
然后，我需要观察这个编号指向的部件的外观和结构特征。

【步骤3：结合上下文】
接下来，我需要结合图片的整体结构、说明书内容和其他信息来判断。

【步骤4：给出答案】
最后，我可以确定这个部件是什么。

现在请开始你的分析，并在最后用【最终答案】标注你的结论（答案通常20字以内）："""
    return prompt


# ============================================
# 答案提取（与test_qwen3_vl.py完全一致）
# ============================================

def extract_final_answer_from_cot(cot_response):
    """从COT响应中提取最终答案"""
    # 方法1: 提取【最终答案】标记的内容
    answer_match = re.search(r'【最终答案】[：:\s]*(.*?)(?:\n|$)', cot_response, re.DOTALL)
    if answer_match:
        raw_answer = answer_match.group(1).strip()
        # 清理可能的多余内容
        raw_answer = raw_answer.split('\n')[0].strip()
        return raw_answer

    # 方法2: 提取最后一个完整句子
    sentences = re.split(r'[。！？]', cot_response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        return sentences[-1]

    return cot_response.strip()


def get_final_answer(text, answer_style):
    """从详细回答中提取简洁答案（与test_qwen3_vl.py完全一致）"""
    question = "你是一个内容提取专家，请从文本中判断，这段描述想表达的准确答案是什么。请仔细思考，在思考结束后，输出简要的答案（通常20个字词以内）。"
    question += """
    为了便于你回答，我给你提供几个示例：
    示例1
    文本内容为："根据专利内容和图2的描述：\n\n- 编号为15的部件是**滤网**。\n- 编号为12的部件是**连接法兰**。\n\n从图2中可以看出，滤网（15）位于连接法兰（12）的**下方**。\n\n因此，正确答案是在12的下方"
    输出的答案为：在12的下方

    示例2
    文本内容为："根据专利内容，调节可移动折弯模架的位置是通过丝杆机构（部件7）实现的。丝杆机构包括丝杆（71）和丝杆滑块（72），通过调节手轮（74）转动丝杆，从而带动丝杆滑块移动，进而控制导轨滑块（6）沿导轨（5）移动，最终实现可移动折弯模架（1）的位置调节。因此，首先需要操作的部件是丝杆机构（部件7）。"
    输出的答案为：部件7

    示例3
    文本内容为："该专利提供了一种用于滚筒输送机的货物靠边规整处理机构，通过倾斜设置的转辊和联动皮带，实现货物自动靠边规整，减少损伤，提高输送效率。"
    输出的答案为：实现货物自动靠边规整

    示例4:
    文本内容为："定位杆"
    输出的答案为： 定位杆

    示例5:
    文本内容为："部件22位于部件23的左侧"
    输出的答案为： 部件22位于部件23的左侧
    """
    question += '同时为了便于你回答，我再给你提供一些答案的风格示例：\n'
    question += answer_style + '\n'
    question += "你要判断的文本内容为：\n"
    question += text
    question += "请直接回答文本想要表达的准确答案（风格和前面的示例类似，通常20个字词以内，并且不要改变原始回答的意思），不要解释，你输出的答案为："

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question}
        ]
    }]
    return origin_vllm(messages)


# ============================================
# 问答函数（与test_qwen3_vl.py完全一致，支持COT开关）
# ============================================

def get_image_answer(document_name, question, question_idx, question_type='other', use_cot=False):
    """普通问题的推理（不含特定页码）"""
    question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += "专利内容为：\n"

    retrived_page_list = get_similar_image_embedding(document_name, question_idx, 2, -1)
    retrived_page_num = sorted(retrived_page_list)

    retrived_list = []
    for page_num in retrived_page_num:
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(page_num) + '.jpg'
        if os.path.exists(image_file):
            retrived_list.append(image_file)

    # COT模式：对特定问题类型使用COT prompt
    if use_cot and question_type == 'position':
        # 使用COT prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "专利内容如下：\n"},
            ]
        }]
        for img_file in retrived_list:
            messages[0]['content'].append({
                "type": "image",
                "image": img_file,
                "max_pixels": 602112
            })
        messages[0]['content'].append({
            "type": "text",
            "text": "\n\n" + build_cot_prompt_position(question)
        })
        return origin_vllm(messages, 1024)

    elif use_cot and question_type == 'identification':
        # 使用COT prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "专利内容如下：\n"},
            ]
        }]
        for img_file in retrived_list:
            messages[0]['content'].append({
                "type": "image",
                "image": img_file,
                "max_pixels": 602112
            })
        messages[0]['content'].append({
            "type": "text",
            "text": "\n\n" + build_cot_prompt_identification(question)
        })
        return origin_vllm(messages, 1024)

    else:
        # 原始prompt（与test_qwen3_vl.py完全一致）
        question2 = "\n\n请你在分析专利内容后，回答我的问题：\n"
        question2 += "【我的问题】【" + question + "】\n"
        question2 += "请仔细思考，在思考结束后，请直接给出你的答案："

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question1},
            ]
        }]

        for img_file in retrived_list:
            messages[0]['content'].append({
                "type": "image",
                "image": img_file,
                "max_pixels": 602112
            })

        messages[0]['content'].append({
            "type": "text",
            "text": question2
        })

        return origin_vllm(messages, 2000)


def get_mix_answer_img(document_name, pic_page_num, question, question_idx, if_need_other=True, question_type='other', use_cot=False):
    """含特定页码的问题推理"""
    question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += "该问题针对于这页专利内容里面的图进行提问：\n"

    retrived_page_list = get_similar_image_embedding(document_name, question_idx, 2, pic_page_num)
    retrived_page_num = sorted(retrived_page_list)

    retrived_list = []
    for page_num in retrived_page_num:
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(page_num) + '.jpg'
        if os.path.exists(image_file):
            retrived_list.append(image_file)

    main_image = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(pic_page_num) + '.jpg'

    if not os.path.exists(main_image):
        return "无法找到指定页面的图片"

    # COT模式：对特定问题类型使用COT prompt
    if use_cot and question_type == 'position':
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "该问题针对于这页专利内容里面的图进行提问：\n"},
                {
                    "type": "image",
                    "image": main_image,
                    "max_pixels": 602112
                },
            ]
        }]
        if if_need_other and len(retrived_list) > 0:
            messages[0]['content'].append({"type": "text", "text": "\n\n其他的相关专利内容为：\n"})
            for img_file in retrived_list:
                messages[0]['content'].append({
                    "type": "image",
                    "image": img_file,
                    "max_pixels": 602112
                })
        messages[0]['content'].append({
            "type": "text",
            "text": "\n\n" + build_cot_prompt_position(question)
        })
        return origin_vllm(messages, 1024)

    elif use_cot and question_type == 'identification':
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "该问题针对于这页专利内容里面的图进行提问：\n"},
                {
                    "type": "image",
                    "image": main_image,
                    "max_pixels": 602112
                },
            ]
        }]
        if if_need_other and len(retrived_list) > 0:
            messages[0]['content'].append({"type": "text", "text": "\n\n其他的相关专利内容为：\n"})
            for img_file in retrived_list:
                messages[0]['content'].append({
                    "type": "image",
                    "image": img_file,
                    "max_pixels": 602112
                })
        messages[0]['content'].append({
            "type": "text",
            "text": "\n\n" + build_cot_prompt_identification(question)
        })
        return origin_vllm(messages, 1024)

    else:
        # 原始prompt（与test_qwen3_vl.py完全一致）
        question2 = "\n\n其他的相关专利内容为：\n"
        question3 = "\n\n请你在分析专利内容后，回答我的问题：\n"
        question3 += "【我的问题】【" + question + "】\n"

        # 位置问题特殊处理（与test_qwen3_vl.py一致）
        if "位置" in question and if_need_other:
            question3 += "请仔细思考，在思考结束后，请直接给出你的答案："
        elif "位置" in question:
            question3 += "请仔细思考，你需要特别注意，图中部件的上下、前后、左右位置判断应以标号线所指代的实际结构为准，而不是仅凭直观看数字。"
            question3 += "在思考结束后，请直接给出你的答案："
        else:
            question3 += "请仔细思考，在思考结束后，请直接给出你的答案："

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question1},
                {
                    "type": "image",
                    "image": main_image,
                    "max_pixels": 602112
                },
            ]
        }]

        if if_need_other and len(retrived_list) > 0:
            messages[0]['content'].append({"type": "text", "text": question2})
            for img_file in retrived_list:
                messages[0]['content'].append({
                    "type": "image",
                    "image": img_file,
                    "max_pixels": 602112
                })

        messages[0]['content'].append({
            "type": "text",
            "text": question3
        })

        return origin_vllm(messages, 768)


def classify_question(text):
    """判断问题是否可以直接通过看图回答（与test_qwen3_vl.py完全一致）"""
    question = """你是一个内容分类专家，请判断用户的这个问题能否直接通过看图回答，还是需要参考其他的相关信息来回答。
    判断规则：已知图是结构图，里面只有部件序号，没有部件名称。如果用户的问题是要通过看图判断某些部件的位置关系，这类问题可以直接通过看图回答；如果用户问题涉及到询问部件是什么、部件名称功能和原理等，这些问题需要参考其他的相关信息来回答。
    对于只需要看图回答的问题，请回答字母"Y"；对于需要参考其他的相关信息来回答的问题，请回答字母"N"。

    给你提供一些示例
    示例1： 在文件中第5页提供的图片中，编号为4的部件是什么？
    解析：询问部件名称，图里面是没有的
    回答：N

    示例2:基于文件中第6页的图片，部件4位于哪个部件的延伸方向上？
    解析：询问部件位置关系，图里面是可以看出来的
    回答：Y

    示例3:根据文件中第7页的图片，部件41位于部件3的什么位置？
    解析：询问部件位置关系，图里面是可以看出来的
    回答：Y

    你要判断的用户问题是：
    """
    question += text + "\n"
    question += "请直接回答分类结果，不要解释，你的答案为："

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question}
        ]
    }]
    return origin_vllm(messages, 2000)


# ============================================
# 主推理循环
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL COT推理测试')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--end', type=int, default=500, help='结束索引（默认500条）')
    parser.add_argument('--all', action='store_true', help='测试全部数据')
    parser.add_argument('--no_cot', action='store_true', help='关闭COT（用于对比实验，与test_qwen3_vl.py完全一致）')
    parser.add_argument('--output', type=str, default=None, help='输出文件名')
    args = parser.parse_args()

    use_cot = not args.no_cot

    if args.output:
        output_file = args.output
    else:
        output_file = 'test_qwen3_vl_cot_results.jsonl' if use_cot else 'test_qwen3_vl_no_cot_results.jsonl'

    start_idx = args.start
    end_idx = len(df_question) if args.all else min(args.end, len(df_question))

    # 清空已有文件
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n开始推理，范围: [{start_idx}, {end_idx})")
    print(f"COT模式: {'开启' if use_cot else '关闭（与test_qwen3_vl.py完全一致）'}")
    print(f"结果将保存到: {output_file}")
    print("=" * 60)

    # 统计
    cot_count = {'position': 0, 'identification': 0, 'other': 0}

    for i in trange(start_idx, end_idx, desc="推理进度"):
        question = df_question.loc[i, 'question']
        document_name = df_question.loc[i, 'document']
        question_type_classify = ''  # 用于classify_question的结果
        if_need_other = True
        answer = ''
        style_answer = ''

        # 分类问题类型（用于COT判断）
        question_type = classify_question_type(question)
        cot_count[question_type] += 1

        # 获取相似问题的答案风格（与test_qwen3_vl.py一致）
        answer_style = get_options_for_similar_answer(get_similar_question_embedding(i, 2))

        if "第" in question and "页" in question and "图":
            # 问题含有特定页码
            page_match = re.findall(r"第(\d+)页", question)
            if page_match:
                pic_page_num = int(page_match[0])

                # 使用classify_question判断是否需要其他信息（与test_qwen3_vl.py一致）
                question_type_classify = classify_question(question)

                if 'Y' in question_type_classify or 'y' in question_type_classify:
                    if_need_other = False
                else:
                    if_need_other = True

                answer = get_mix_answer_img(
                    document_name, pic_page_num, question, i,
                    if_need_other, question_type, use_cot
                )
            else:
                answer = get_image_answer(document_name, question, i, question_type, use_cot)
        else:
            # 普通问题
            answer = get_image_answer(document_name, question, i, question_type, use_cot)

        # 提取最终答案
        if use_cot and question_type in ['position', 'identification']:
            # COT模式：先提取【最终答案】，再用风格精炼
            extracted_answer = extract_final_answer_from_cot(answer)
            style_answer = get_final_answer(extracted_answer, answer_style)
        else:
            # 非COT模式：与test_qwen3_vl.py完全一致
            style_answer = get_final_answer(answer, answer_style)

        # 保存结果（格式与test_qwen3_vl.py一致）
        result_dict = {
            'idx': str(i),
            'document': document_name,
            'question': question,
            'question_type': question_type_classify,  # 与test_qwen3_vl.py一致
            'answer': answer,
            'style_answer': style_answer
        }

        # COT模式额外添加字段
        if use_cot:
            result_dict['cot_question_type'] = question_type  # position/identification/other
            result_dict['use_cot'] = question_type in ['position', 'identification']

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(f"推理完成！结果已保存到: {output_file}")
    if use_cot:
        print(f"\n问题类型统计:")
        print(f"  位置关系问题: {cot_count['position']} (使用COT)")
        print(f"  部件识别问题: {cot_count['identification']} (使用COT)")
        print(f"  其他问题: {cot_count['other']} (普通推理)")
    print("=" * 60)


if __name__ == "__main__":
    main()
