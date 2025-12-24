# -*- coding: utf-8 -*-
"""
位置类问题数据增强脚本
针对部件位置识别效果差的问题，筛选并增强位置类问题数据

使用方法:
    python position_data_augmentation.py \
        --train_jsonl /data/coding/patent_b/train/train.jsonl \
        --img_base_dir /data/coding/patent_b/train/pdf_img \
        --output_dir /data/coding/ \
        --output_name train_b_position_augmented.jsonl
"""

import json
import re
import os
import argparse
from typing import List, Dict, Optional


# ============== 位置类问题关键词 ==============
POSITION_KEYWORDS = [
    '位置关系', '位置是', '什么位置', '哪个位置', '哪个方位',
    '上方', '下方', '左侧', '右侧', '左边', '右边',
    '前面', '后面', '前方', '后方',
    '正上方', '正下方', '正左方', '正右方',
    '上侧', '下侧', '左上', '右上', '左下', '右下',
    '相邻', '旁边', '对侧', '对面'
]


def is_position_question(question: str) -> bool:
    """判断是否为位置类问题"""
    return any(kw in question for kw in POSITION_KEYWORDS)


def is_image_question(question: str) -> bool:
    """判断是否涉及图片分析"""
    return '页' in question and ('图' in question or '示意图' in question or '图片' in question)


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


def load_train_data(path: str) -> List[Dict]:
    """加载训练数据"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def filter_position_questions(data: List[Dict]) -> List[Dict]:
    """筛选位置类问题"""
    position_data = []
    for item in data:
        question = item['question']
        if is_position_question(question):
            item['is_image_question'] = is_image_question(question)
            item['page_number'] = extract_page_number(question)
            item['component_numbers'] = extract_component_numbers(question)
            position_data.append(item)
    return position_data


# ============== Prompt模板 ==============

def create_prompt_v1(question: str) -> str:
    """增强策略1: 标号线追踪提示"""
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


def create_prompt_v2(question: str) -> str:
    """增强策略2: CoT思考链"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

请按以下步骤仔细分析图中部件的位置关系：

步骤1：在图中找到问题涉及的所有编号标签
步骤2：对每个编号，追踪其标号线（引出线/指示线）到达的终点
步骤3：确定标号线终点所指向的实际部件/结构的位置
步骤4：比较各部件在图中的实际空间位置关系
步骤5：根据图片方向（上下左右）给出准确的位置描述

【我的问题】【{question}】

请根据上述分析步骤，给出你的答案："""


def create_prompt_v3(question: str) -> str:
    """增强策略3: 简洁强调版"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

【关键提示】图中部件的位置判断应以标号线所指代的实际结构为准，而不是仅凭数字标签的位置。

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


def create_prompt_v4_with_context(question: str) -> str:
    """增强策略4: 带额外上下文图片"""
    return f"""你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。
该问题针对于这页专利内容里面的图进行提问：
<image>

其他的相关专利内容为：
<image><image>

【重要提示】判断部件位置时，请以标号线指向的实际结构为准，而非数字本身的位置。

请你在分析专利内容后，回答我的问题：
【我的问题】【{question}】
请仔细思考，在思考结束后，请直接给出你的答案："""


def generate_augmented_samples(
    position_data: List[Dict],
    img_base_dir: str,
    use_all_strategies: bool = True
) -> List[Dict]:
    """
    生成增强样本

    Args:
        position_data: 位置类问题数据
        img_base_dir: 图片基础目录
        use_all_strategies: 是否使用所有增强策略
    """
    augmented_samples = []

    for idx, item in enumerate(position_data):
        question = item['question']
        answer = item['answer']
        document = item['document']
        page_number = item.get('page_number')
        component_numbers = item.get('component_numbers', [])

        # 只处理有页码的图片类问题
        if not page_number:
            continue

        doc_name = document.replace('.pdf', '')
        main_image = f"{img_base_dir}/{doc_name}/{page_number}.jpg"

        # 策略1: 标号线追踪提示 (所有位置问题)
        sample_v1 = {
            'query': create_prompt_v1(question),
            'document': document,
            'question_idx': item.get('id', idx),
            'response': answer,
            'images': [main_image],
            'augment_type': 'v1_tracking'
        }
        augmented_samples.append(sample_v1)

        if use_all_strategies:
            # 策略2: CoT思考链 (涉及多个部件的问题)
            if len(component_numbers) >= 2:
                sample_v2 = {
                    'query': create_prompt_v2(question),
                    'document': document,
                    'question_idx': item.get('id', idx),
                    'response': answer,
                    'images': [main_image],
                    'augment_type': 'v2_cot'
                }
                augmented_samples.append(sample_v2)

            # 策略3: 简洁强调版 (所有位置问题)
            sample_v3 = {
                'query': create_prompt_v3(question),
                'document': document,
                'question_idx': item.get('id', idx),
                'response': answer,
                'images': [main_image],
                'augment_type': 'v3_concise'
            }
            augmented_samples.append(sample_v3)

    return augmented_samples


def analyze_and_report(train_data: List[Dict], position_data: List[Dict]):
    """分析并输出报告"""
    print("=" * 60)
    print("位置类问题分析报告")
    print("=" * 60)

    total_train = len(train_data)
    total_position = len(position_data)
    image_questions = sum(1 for item in position_data if item.get('is_image_question'))

    print(f"训练集总问题数: {total_train}")
    print(f"位置类问题数: {total_position} ({total_position/total_train*100:.1f}%)")
    print(f"涉及图片的位置问题: {image_questions} ({image_questions/total_position*100:.1f}%)")

    # 关键词分布
    print("\n关键词分布 (Top 10):")
    keyword_counts = {}
    for item in position_data:
        q = item['question']
        for kw in POSITION_KEYWORDS:
            if kw in q:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kw}: {count}")

    # 按文档统计
    print("\n按文档分布 (Top 10):")
    doc_counts = {}
    for item in position_data:
        doc = item['document']
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    for doc, count in sorted(doc_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {doc}: {count}")

    # 有多个部件编号的问题
    multi_component = sum(1 for item in position_data if len(item.get('component_numbers', [])) >= 2)
    print(f"\n涉及多个部件的问题: {multi_component}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='位置类问题数据增强')
    parser.add_argument('--train_jsonl', type=str, required=True,
                        help='训练集jsonl路径')
    parser.add_argument('--img_base_dir', type=str, required=True,
                        help='PDF图片目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--output_name', type=str, default='train_position_augmented.jsonl',
                        help='输出文件名')
    parser.add_argument('--all_strategies', action='store_true', default=True,
                        help='使用所有增强策略')
    parser.add_argument('--analyze_only', action='store_true',
                        help='只分析不生成数据')

    args = parser.parse_args()

    # 加载数据
    print(f"加载训练数据: {args.train_jsonl}")
    train_data = load_train_data(args.train_jsonl)
    print(f"总样本数: {len(train_data)}")

    # 筛选位置类问题
    print("\n筛选位置类问题...")
    position_data = filter_position_questions(train_data)
    print(f"位置类问题数: {len(position_data)}")

    # 分析报告
    analyze_and_report(train_data, position_data)

    if args.analyze_only:
        print("\n[分析模式] 不生成数据")
        return

    # 生成增强数据
    print("\n生成增强数据...")
    augmented_samples = generate_augmented_samples(
        position_data,
        args.img_base_dir,
        use_all_strategies=args.all_strategies
    )
    print(f"生成增强样本数: {len(augmented_samples)}")

    # 统计各策略样本数
    strategy_counts = {}
    for sample in augmented_samples:
        aug_type = sample.get('augment_type', 'unknown')
        strategy_counts[aug_type] = strategy_counts.get(aug_type, 0) + 1

    print("\n各策略样本数:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")

    # 保存
    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            # 移除augment_type字段（训练时不需要）
            sample_out = {k: v for k, v in sample.items() if k != 'augment_type'}
            f.write(json.dumps(sample_out, ensure_ascii=False) + '\n')

    print(f"\n保存到: {output_path}")

    # 打印示例
    print("\n" + "=" * 60)
    print("增强样本示例:")
    print("=" * 60)
    if augmented_samples:
        sample = augmented_samples[0]
        print(f"Query:\n{sample['query']}")
        print(f"\nResponse: {sample['response']}")
        print(f"Images: {sample['images']}")


if __name__ == '__main__':
    main()
