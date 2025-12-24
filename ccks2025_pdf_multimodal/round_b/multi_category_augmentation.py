# -*- coding: utf-8 -*-
"""
多类别问题数据增强脚本
根据模型在各类别上的表现，针对性地增强低分类别

各类别得分：
- 工作原理: 0.844 ✅ 效果好
- 数量相关: 0.600
- 其他: 0.500
- 功能作用: 0.489 ⭐ 需增强
- 部件识别: 0.480 ⭐ 需增强
- 指定页面: 0.471 ⭐ 需增强
- 位置关系: 0.448 ⭐⭐ 重点增强
- 连接关系: 0.380 ⭐⭐⭐ 最需增强

使用方法:
    python multi_category_augmentation.py \
        --train_jsonl /data/coding/patent_b/train/train.jsonl \
        --img_base_dir /data/coding/patent_b/train/pdf_img \
        --output_dir /data/coding/ \
        --output_name train_b_multi_augmented.jsonl
"""

import json
import re
import os
import argparse
from typing import List, Dict, Optional, Tuple


# ============== 问题分类（与 analyze_model_diff.py 一致） ==============

def classify_question_category(question: str) -> List[str]:
    """对问题进行更细粒度的分类"""
    categories = []

    # 位置关系
    position_keywords = ['位置', '上方', '下方', '左侧', '右侧', '前方', '后方',
                         '旁边', '之间', '内部', '外部', '顶部', '底部', '哪里']
    if any(kw in question for kw in position_keywords):
        categories.append('位置关系')

    # 部件识别
    identification_keywords = ['是什么', '什么部件', '哪个部件', '叫什么', '编号为']
    if any(kw in question for kw in identification_keywords):
        categories.append('部件识别')

    # 功能作用
    function_keywords = ['功能', '作用', '用于', '用来', '目的', '为了']
    if any(kw in question for kw in function_keywords):
        categories.append('功能作用')

    # 连接关系
    connection_keywords = ['连接', '固定', '安装', '配合', '接触']
    if any(kw in question for kw in connection_keywords):
        categories.append('连接关系')

    # 工作原理
    principle_keywords = ['原理', '如何工作', '怎样', '如何实现', '过程']
    if any(kw in question for kw in principle_keywords):
        categories.append('工作原理')

    # 特定页面图片
    if '第' in question and '页' in question:
        categories.append('指定页面')

    # 数量相关
    if any(kw in question for kw in ['多少', '几个', '数量']):
        categories.append('数量相关')

    if not categories:
        categories.append('其他')

    return categories


def extract_page_number(question: str) -> Optional[int]:
    """从问题中提取页码"""
    match = re.search(r'第(\d+)页', question)
    if match:
        return int(match.group(1))
    return None


def extract_component_numbers(question: str) -> List[str]:
    """从问题中提取部件编号"""
    patterns = [
        r'编号为(\d+)',
        r'编号(\d+)',
        r'部件(\d+)',
        r'\((\d+)\)',
        r'（(\d+)）',
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, question)
        numbers.extend(matches)
    return list(set(numbers))


# ============== 各类别专用 Prompt 模板 ==============

# ---------- 位置关系 Prompt ----------
def prompt_position_v1(question: str) -> str:
    """位置关系-标号线追踪"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【重要分析步骤】
1. 首先找到问题中提到的每个编号对应的标号线（引出线）
2. 仔细追踪每条标号线，确定它指向的实际部件结构位置
3. 注意：判断位置关系时，以标号线指向的部件结构为准，而非数字标签本身的位置
4. 上下左右方位以图片的视觉方向为准

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


def prompt_position_v2(question: str) -> str:
    """位置关系-CoT步骤"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

请按以下步骤分析图中部件的位置关系：
步骤1：在图中找到问题涉及的所有编号标签
步骤2：追踪每个编号的标号线到达的终点，确定实际部件位置
步骤3：比较各部件在图中的实际空间位置关系
步骤4：根据图片方向（上下左右）给出准确的位置描述

【我的问题】【{question}】
请根据上述分析步骤，给出你的答案："""


# ---------- 连接关系 Prompt ----------
def prompt_connection_v1(question: str) -> str:
    """连接关系-结构分析"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【连接关系分析提示】
1. 仔细观察图中各部件之间的连接方式（螺栓连接、焊接、卡接、套接等）
2. 注意部件之间的接触面和连接点
3. 追踪标号线确定每个编号对应的具体部件
4. 分析部件之间的装配顺序和连接逻辑

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


def prompt_connection_v2(question: str) -> str:
    """连接关系-CoT步骤"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

请按以下步骤分析部件的连接关系：
步骤1：找到问题中提到的部件编号，追踪标号线确定具体部件
步骤2：观察该部件与周围部件的接触点和连接方式
步骤3：判断是直接连接还是通过其他部件间接连接
步骤4：如涉及固定/安装方式，分析具体的固定结构

【我的问题】【{question}】
请根据上述分析步骤，给出你的答案："""


# ---------- 部件识别 Prompt ----------
def prompt_identification_v1(question: str) -> str:
    """部件识别-标号追踪"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【部件识别提示】
1. 找到问题中提到的编号在图中的位置
2. 仔细追踪该编号的标号线（引出线），确定它指向的具体部件
3. 根据部件的形状、位置和功能特征进行识别
4. 如图中有说明书文字，参考文字描述确认部件名称

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


def prompt_identification_v2(question: str) -> str:
    """部件识别-特征分析"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

请按以下步骤识别部件：
步骤1：在图中定位问题提到的编号
步骤2：追踪标号线找到实际指向的部件结构
步骤3：观察该部件的形状特征（圆形、方形、杆状等）
步骤4：结合部件在整体结构中的位置和功能推断其名称

【我的问题】【{question}】
请根据上述分析步骤，给出你的答案："""


# ---------- 功能作用 Prompt ----------
def prompt_function_v1(question: str) -> str:
    """功能作用-综合分析"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【功能分析提示】
1. 找到问题涉及的部件，观察其结构特征
2. 分析该部件与其他部件的连接和配合关系
3. 结合整体装置的工作流程，推断该部件的功能
4. 如有说明书文字，参考文字中对该部件功能的描述

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


# ---------- 指定页面 Prompt ----------
def prompt_specified_page_v1(question: str) -> str:
    """指定页面-图文结合"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【图片分析提示】
1. 仔细观察图中的所有标注编号和标号线
2. 每个标号线指向的位置才是该编号代表的实际部件
3. 注意区分数字标签位置和标号线终点位置
4. 结合图中的文字说明（如有）进行综合判断

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


# ============== 数据加载和处理 ==============

def load_train_data(path: str) -> List[Dict]:
    """加载训练数据"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_sample(
    question: str,
    answer: str,
    document: str,
    page_number: int,
    img_base_dir: str,
    prompt_func,
    augment_type: str,
    question_id: int
) -> Dict:
    """创建训练样本"""
    doc_name = document.replace('.pdf', '')
    main_image = f"{img_base_dir}/{doc_name}/{page_number}.jpg"

    return {
        'query': prompt_func(question),
        'document': document,
        'question_idx': question_id,
        'response': answer,
        'images': [main_image],
        'augment_type': augment_type
    }


def generate_augmented_samples(
    train_data: List[Dict],
    img_base_dir: str,
    augment_config: Dict[str, int]
) -> Tuple[List[Dict], Dict]:
    """
    生成增强样本

    Args:
        train_data: 训练数据
        img_base_dir: 图片目录
        augment_config: 各类别增强倍数配置

    Returns:
        augmented_samples: 增强样本列表
        stats: 统计信息
    """
    augmented_samples = []
    stats = {cat: {'original': 0, 'augmented': 0} for cat in augment_config.keys()}
    stats['total'] = {'original': 0, 'augmented': 0}

    for item in train_data:
        question = item['question']
        answer = item['answer']
        document = item['document']
        question_id = item.get('id', 0)

        categories = classify_question_category(question)
        page_number = extract_page_number(question)

        # 统计原始数据
        for cat in categories:
            if cat in stats:
                stats[cat]['original'] += 1
        stats['total']['original'] += 1

        # 如果没有页码，跳过（无法定位图片）
        if not page_number:
            continue

        # 根据类别生成增强样本
        for cat in categories:
            if cat not in augment_config:
                continue

            multiplier = augment_config[cat]
            if multiplier <= 0:
                continue

            # 根据类别选择prompt
            if cat == '位置关系':
                prompts = [
                    (prompt_position_v1, 'position_v1'),
                    (prompt_position_v2, 'position_v2'),
                ]
            elif cat == '连接关系':
                prompts = [
                    (prompt_connection_v1, 'connection_v1'),
                    (prompt_connection_v2, 'connection_v2'),
                ]
            elif cat == '部件识别':
                prompts = [
                    (prompt_identification_v1, 'identification_v1'),
                    (prompt_identification_v2, 'identification_v2'),
                ]
            elif cat == '功能作用':
                prompts = [
                    (prompt_function_v1, 'function_v1'),
                ]
            elif cat == '指定页面':
                prompts = [
                    (prompt_specified_page_v1, 'specified_page_v1'),
                ]
            else:
                continue

            # 生成增强样本
            for prompt_func, aug_type in prompts[:multiplier]:
                sample = create_sample(
                    question, answer, document, page_number,
                    img_base_dir, prompt_func, aug_type, question_id
                )
                augmented_samples.append(sample)
                stats[cat]['augmented'] += 1
                stats['total']['augmented'] += 1

    return augmented_samples, stats


def analyze_and_report(train_data: List[Dict]):
    """分析训练数据分布"""
    print("=" * 60)
    print("训练数据类别分布分析")
    print("=" * 60)

    category_counts = {}
    for item in train_data:
        categories = classify_question_category(item['question'])
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n总样本数: {len(train_data)}")
    print("\n各类别问题数量:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(train_data)*100:.1f}%)")

    return category_counts


def main():
    parser = argparse.ArgumentParser(description='多类别问题数据增强')
    parser.add_argument('--train_jsonl', type=str, required=True,
                        help='训练集jsonl路径')
    parser.add_argument('--img_base_dir', type=str, required=True,
                        help='PDF图片目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--output_name', type=str, default='train_multi_augmented.jsonl',
                        help='输出文件名')
    parser.add_argument('--analyze_only', action='store_true',
                        help='只分析不生成数据')

    # 各类别增强倍数（根据得分调整，得分越低增强越多）
    parser.add_argument('--aug_connection', type=int, default=2,
                        help='连接关系增强倍数 (得分0.38，最需增强)')
    parser.add_argument('--aug_position', type=int, default=2,
                        help='位置关系增强倍数 (得分0.448)')
    parser.add_argument('--aug_identification', type=int, default=2,
                        help='部件识别增强倍数 (得分0.48)')
    parser.add_argument('--aug_function', type=int, default=1,
                        help='功能作用增强倍数 (得分0.489)')
    parser.add_argument('--aug_specified_page', type=int, default=1,
                        help='指定页面增强倍数 (得分0.471)')

    args = parser.parse_args()

    # 增强配置
    augment_config = {
        '连接关系': args.aug_connection,      # 0.380 - 最需增强
        '位置关系': args.aug_position,        # 0.448
        '部件识别': args.aug_identification,  # 0.480
        '功能作用': args.aug_function,        # 0.489
        '指定页面': args.aug_specified_page,  # 0.471
    }

    # 加载数据
    print(f"加载训练数据: {args.train_jsonl}")
    train_data = load_train_data(args.train_jsonl)
    print(f"总样本数: {len(train_data)}")

    # 分析
    category_counts = analyze_and_report(train_data)

    if args.analyze_only:
        print("\n[分析模式] 不生成数据")
        return

    # 生成增强数据
    print("\n" + "=" * 60)
    print("生成增强数据...")
    print("=" * 60)

    print("\n增强配置:")
    for cat, mult in augment_config.items():
        print(f"  {cat}: x{mult}")

    augmented_samples, stats = generate_augmented_samples(
        train_data, args.img_base_dir, augment_config
    )

    print(f"\n生成统计:")
    print(f"  原始样本: {stats['total']['original']}")
    print(f"  增强样本: {stats['total']['augmented']}")
    print(f"\n各类别增强情况:")
    for cat in augment_config.keys():
        if cat in stats:
            print(f"  {cat}: 原始{stats[cat]['original']} -> 增强{stats[cat]['augmented']}")

    # 统计各增强类型
    aug_type_counts = {}
    for sample in augmented_samples:
        aug_type = sample.get('augment_type', 'unknown')
        aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

    print(f"\n各Prompt类型样本数:")
    for aug_type, count in sorted(aug_type_counts.items()):
        print(f"  {aug_type}: {count}")

    # 保存
    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            sample_out = {k: v for k, v in sample.items() if k != 'augment_type'}
            f.write(json.dumps(sample_out, ensure_ascii=False) + '\n')

    print(f"\n保存到: {output_path}")

    # 打印示例
    print("\n" + "=" * 60)
    print("增强样本示例:")
    print("=" * 60)

    # 每种类型打印一个示例
    shown_types = set()
    for sample in augmented_samples:
        aug_type = sample.get('augment_type', '')
        if aug_type not in shown_types:
            shown_types.add(aug_type)
            print(f"\n【{aug_type}】")
            print(f"Query:\n{sample['query'][:300]}...")
            print(f"Response: {sample['response']}")
            if len(shown_types) >= 3:
                break


if __name__ == '__main__':
    main()
