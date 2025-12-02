#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL 模型消融实验脚本
用于测试各模块对准确度的贡献

通过配置开关控制各模块是否启用
"""

import numpy as np
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import re
import json
from tqdm import trange
import argparse

# ============================================
# 消融实验配置
# ============================================

class AblationConfig:
    """消融实验配置"""
    def __init__(self):
        # 模块开关
        self.use_question_classify = True      # 是否使用问题分类
        self.use_image_retrieval = True        # 是否使用图片检索（关闭则只用问题指定的页面）
        self.use_answer_style = True           # 是否使用答案风格匹配
        self.use_answer_extract = True         # 是否使用答案提炼
        self.use_position_hint = True          # 是否使用位置问题特殊提示

        # 模型选择
        self.use_finetuned = True              # True: 使用微调后的模型, False: 使用原始模型

        # 测试范围
        self.test_start = 0                    # 起始索引
        self.test_end = None                   # 结束索引（None表示全部）

        # 实验名称（用于区分输出文件）
        self.exp_name = "full"

    def get_output_file(self):
        return f'ablation_{self.exp_name}_results.jsonl'

    def get_model_path(self):
        if self.use_finetuned:
            return "/usr/yuque/guo/pdf_processer/qwen3_vl_32b_merged"
        else:
            return "/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen3-VL-32B-Instruct"

    def __str__(self):
        return f"""
消融实验配置:
  - 模型: {'微调后' if self.use_finetuned else '原始模型'}
  - 问题分类: {'开启' if self.use_question_classify else '关闭'}
  - 图片检索: {'开启' if self.use_image_retrieval else '关闭'}
  - 答案风格匹配: {'开启' if self.use_answer_style else '关闭'}
  - 答案提炼: {'开启' if self.use_answer_extract else '关闭'}
  - 位置特殊提示: {'开启' if self.use_position_hint else '关闭'}
  - 测试范围: [{self.test_start}, {self.test_end or '全部'}]
  - 实验名称: {self.exp_name}
"""


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

# 全局变量，延迟加载模型
vl_model = None
processor = None
current_model_path = None


def load_model(model_path):
    """加载模型（如果路径不同才重新加载）"""
    global vl_model, processor, current_model_path

    if current_model_path == model_path and vl_model is not None:
        print(f"模型已加载: {model_path}")
        return

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
    current_model_path = model_path

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
# 问答函数（支持消融配置）
# ============================================

def get_image_answer(document_name, question, question_idx, config):
    """普通问题的推理（不含特定页码）"""
    question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += "专利内容为：\n"

    # 根据配置决定是否使用图片检索
    if config.use_image_retrieval:
        retrived_page_list = get_similar_image_embedding(document_name, question_idx, 2, -1)
    else:
        # 不使用检索，使用默认的前2页
        retrived_page_list = [1, 2]

    retrived_page_num = sorted(retrived_page_list)

    retrived_list = []
    for page_num in retrived_page_num:
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(page_num) + '.jpg'
        if os.path.exists(image_file):
            retrived_list.append(image_file)

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


def get_mix_answer_img(document_name, pic_page_num, question, question_idx, if_need_other, config):
    """含特定页码的问题推理"""
    question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += "该问题针对于这页专利内容里面的图进行提问：\n"

    # 根据配置决定是否使用图片检索
    if config.use_image_retrieval and if_need_other:
        retrived_page_list = get_similar_image_embedding(document_name, question_idx, 2, pic_page_num)
        retrived_page_num = sorted(retrived_page_list)
    else:
        retrived_page_num = []

    retrived_list = []
    for page_num in retrived_page_num:
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(page_num) + '.jpg'
        if os.path.exists(image_file):
            retrived_list.append(image_file)

    question2 = "\n\n其他的相关专利内容为：\n"
    question3 = "\n\n请你在分析专利内容后，回答我的问题：\n"
    question3 += "【我的问题】【" + question + "】\n"

    # 根据配置决定是否使用位置特殊提示
    if config.use_position_hint and "位置" in question and if_need_other:
        question3 += "请仔细思考，在思考结束后，请直接给出你的答案："
    elif config.use_position_hint and "位置" in question:
        question3 += "请仔细思考，你需要特别注意，图中部件的上下、前后、左右位置判断应以标号线所指代的实际结构为准，而不是仅凭直观看数字。"
        question3 += "在思考结束后，请直接给出你的答案："
    else:
        question3 += "请仔细思考，在思考结束后，请直接给出你的答案："

    main_image = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(pic_page_num) + '.jpg'

    if not os.path.exists(main_image):
        return "无法找到指定页面的图片"

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


def classify_question(text, config):
    """判断问题是否可以直接通过看图回答"""
    if not config.use_question_classify:
        # 不使用分类，默认需要其他信息
        return "N"

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


def get_final_answer(text, answer_style, config):
    """从详细回答中提取简洁答案"""
    if not config.use_answer_extract:
        # 不使用答案提炼，直接返回原始答案
        return text

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

    # 根据配置决定是否使用答案风格匹配
    if config.use_answer_style and answer_style:
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
# 主推理循环
# ============================================

def run_ablation(config):
    """运行消融实验"""
    # 加载对应的模型
    model_path = config.get_model_path()
    load_model(model_path)

    output_file = config.get_output_file()

    # 清空已有文件
    if os.path.exists(output_file):
        os.remove(output_file)

    print(config)
    print(f"结果将保存到: {output_file}")
    print("=" * 60)

    # 确定测试范围
    start_idx = config.test_start
    end_idx = config.test_end if config.test_end else len(df_question)

    for i in trange(start_idx, end_idx, desc="推理进度"):
        question = df_question.loc[i, 'question']
        document_name = df_question.loc[i, 'document']
        question_type = ''
        if_need_other = True
        answer = ''
        style_answer = ''

        # 根据配置决定是否获取答案风格
        if config.use_answer_style:
            answer_style = get_options_for_similar_answer(get_similar_question_embedding(i, 2))
        else:
            answer_style = ''

        if "第" in question and "页" in question and "图":
            page_match = re.findall(r"第(\d+)页", question)
            if page_match:
                pic_page_num = int(page_match[0])
                question_type = classify_question(question, config)

                if 'Y' in question_type or 'y' in question_type:
                    if_need_other = False
                else:
                    if_need_other = True

                answer = get_mix_answer_img(document_name, pic_page_num, question, i, if_need_other, config)
                style_answer = get_final_answer(answer, answer_style, config)
            else:
                answer = get_image_answer(document_name, question, i, config)
                style_answer = get_final_answer(answer, answer_style, config)
        else:
            answer = get_image_answer(document_name, question, i, config)
            style_answer = get_final_answer(answer, answer_style, config)

        # 保存结果
        result_dict = {
            'idx': str(i),
            'document': document_name,
            'question': question,
            'question_type': question_type,
            'answer': answer,
            'style_answer': style_answer,
            'config': config.exp_name
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(f"实验 [{config.exp_name}] 完成！结果已保存到: {output_file}")
    print("=" * 60)


# ============================================
# 预定义的消融实验配置
# ============================================

def get_ablation_configs():
    """返回各种消融实验配置"""
    configs = {}

    # ==========================================
    # 微调后模型的实验
    # ==========================================

    # 1. 完整模型（基线）- 微调后
    full = AblationConfig()
    full.exp_name = "full"
    configs["full"] = full

    # 2. 关闭问题分类
    no_classify = AblationConfig()
    no_classify.use_question_classify = False
    no_classify.exp_name = "no_classify"
    configs["no_classify"] = no_classify

    # 3. 关闭图片检索
    no_retrieval = AblationConfig()
    no_retrieval.use_image_retrieval = False
    no_retrieval.exp_name = "no_retrieval"
    configs["no_retrieval"] = no_retrieval

    # 4. 关闭答案风格匹配
    no_style = AblationConfig()
    no_style.use_answer_style = False
    no_style.exp_name = "no_style"
    configs["no_style"] = no_style

    # 5. 关闭答案提炼
    no_extract = AblationConfig()
    no_extract.use_answer_extract = False
    no_extract.exp_name = "no_extract"
    configs["no_extract"] = no_extract

    # 6. 关闭位置特殊提示
    no_position = AblationConfig()
    no_position.use_position_hint = False
    no_position.exp_name = "no_position"
    configs["no_position"] = no_position

    # 7. 最简模型（只保留基本问答）- 微调后
    minimal = AblationConfig()
    minimal.use_question_classify = False
    minimal.use_image_retrieval = False
    minimal.use_answer_style = False
    minimal.use_answer_extract = False
    minimal.use_position_hint = False
    minimal.exp_name = "minimal"
    configs["minimal"] = minimal

    # ==========================================
    # 原始模型的实验（对比微调效果）
    # ==========================================

    # 8. 原始模型 + 完整pipeline
    base_full = AblationConfig()
    base_full.use_finetuned = False
    base_full.exp_name = "base_full"
    configs["base_full"] = base_full

    # 9. 原始模型 + 最简pipeline
    base_minimal = AblationConfig()
    base_minimal.use_finetuned = False
    base_minimal.use_question_classify = False
    base_minimal.use_image_retrieval = False
    base_minimal.use_answer_style = False
    base_minimal.use_answer_extract = False
    base_minimal.use_position_hint = False
    base_minimal.exp_name = "base_minimal"
    configs["base_minimal"] = base_minimal

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen3-VL 消融实验')
    parser.add_argument('--exp', type=str, default='full',
                        choices=['full', 'no_classify', 'no_retrieval', 'no_style',
                                'no_extract', 'no_position', 'minimal',
                                'base_full', 'base_minimal'],
                        help='选择实验配置')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--end', type=int, default=500, help='结束索引（默认500条）')
    parser.add_argument('--all', action='store_true', help='测试全部数据（忽略--end参数）')

    args = parser.parse_args()

    configs = get_ablation_configs()
    config = configs[args.exp]
    config.test_start = args.start

    # 如果指定了--all，则测试全部数据
    if args.all:
        config.test_end = None
    else:
        config.test_end = args.end

    run_ablation(config)
