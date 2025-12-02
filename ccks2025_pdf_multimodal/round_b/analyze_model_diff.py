#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析不同模型结果的差异
找出各模型表现好/差的样本，分析问题类型特点
"""

import json
import pandas as pd
import numpy as np
import re
import argparse
from collections import Counter
from tqdm import tqdm

# 评估指标函数
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_for_em(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_chinese(text):
    return list(text.replace(' ', ''))

def exact_match(pred, gold):
    return float(normalize_for_em(pred) == normalize_for_em(gold))

def rouge_1_f1(pred, gold):
    pred_tokens = tokenize_chinese(normalize_text(pred))
    gold_tokens = tokenize_chinese(normalize_text(gold))

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = overlap / len(gold_tokens) if len(gold_tokens) > 0 else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def compute_score(pred, gold):
    """计算单个样本的得分（简化版，不用BERTScore）"""
    em = exact_match(pred, gold)
    rouge1 = rouge_1_f1(pred, gold)
    # 简化：用rouge1代替bert_score
    score = 0.5 * rouge1 + 0.25 * rouge1 + 0.25 * em
    return score, em, rouge1


def classify_question_category(question):
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


def load_results(file_path):
    """加载结果文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_gold_answers(gold_file):
    """加载标准答案"""
    gold_dict = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_dict[item['id']] = item['answer']
    return gold_dict


def analyze_single_model(results, gold_dict, model_name):
    """分析单个模型的结果"""
    scores = []

    for item in results:
        idx = int(item['idx'])
        gold_id = idx + 1

        if gold_id not in gold_dict:
            continue

        pred = item.get('style_answer', item.get('answer', ''))
        gold = gold_dict[gold_id]
        question = item.get('question', '')

        score, em, rouge1 = compute_score(pred, gold)
        categories = classify_question_category(question)

        scores.append({
            'idx': idx,
            'question': question,
            'pred': pred,
            'gold': gold,
            'score': score,
            'em': em,
            'rouge1': rouge1,
            'categories': categories,
            'use_cot': item.get('use_cot', False),
            'cot_type': item.get('cot_question_type', 'unknown')
        })

    return pd.DataFrame(scores)


def compare_models(model_results, gold_dict, model_names):
    """对比多个模型的结果"""
    all_dfs = {}

    for name, results in zip(model_names, model_results):
        df = analyze_single_model(results, gold_dict, name)
        all_dfs[name] = df
        print(f"\n{name}: 平均得分 = {df['score'].mean():.4f}, 样本数 = {len(df)}")

    return all_dfs


def find_diff_samples(df1, df2, name1, name2, threshold=0.2):
    """找出两个模型差异较大的样本"""
    merged = df1.merge(df2, on='idx', suffixes=(f'_{name1}', f'_{name2}'))

    # 计算得分差异
    merged['score_diff'] = merged[f'score_{name1}'] - merged[f'score_{name2}']

    # 模型1更好的样本
    better_in_1 = merged[merged['score_diff'] > threshold].sort_values('score_diff', ascending=False)

    # 模型2更好的样本
    better_in_2 = merged[merged['score_diff'] < -threshold].sort_values('score_diff', ascending=True)

    return better_in_1, better_in_2, merged


def analyze_category_performance(df, model_name):
    """分析各类别问题的表现"""
    # 展开categories
    category_scores = {}

    for _, row in df.iterrows():
        for cat in row['categories']:
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(row['score'])

    # 计算统计
    category_stats = []
    for cat, scores in category_scores.items():
        category_stats.append({
            'category': cat,
            'count': len(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        })

    return pd.DataFrame(category_stats).sort_values('mean_score', ascending=False)


def main():
    parser = argparse.ArgumentParser(description='分析模型结果差异')
    parser.add_argument('--results', nargs='+', required=True, help='结果文件列表')
    parser.add_argument('--names', nargs='+', required=True, help='模型名称列表')
    parser.add_argument('--gold', type=str,
                        default='/usr/yuque/guo/pdf_processer/patent_b/test/answers.jsonl',
                        help='标准答案文件')
    parser.add_argument('--output', type=str, default='model_analysis', help='输出文件前缀')
    parser.add_argument('--top_n', type=int, default=20, help='显示前N个差异样本')
    args = parser.parse_args()

    if len(args.results) != len(args.names):
        print("错误: 结果文件数量和名称数量不匹配")
        return

    # 加载数据
    print("加载标准答案...")
    gold_dict = load_gold_answers(args.gold)
    print(f"标准答案数量: {len(gold_dict)}")

    print("\n加载模型结果...")
    model_results = []
    for f in args.results:
        results = load_results(f)
        model_results.append(results)
        print(f"  {f}: {len(results)} 条")

    # 分析每个模型
    print("\n" + "=" * 80)
    print("各模型整体表现")
    print("=" * 80)
    all_dfs = compare_models(model_results, gold_dict, args.names)

    # 分析各类别表现
    print("\n" + "=" * 80)
    print("各类别问题表现分析")
    print("=" * 80)

    category_results = {}
    for name, df in all_dfs.items():
        print(f"\n【{name}】各类别表现:")
        cat_stats = analyze_category_performance(df, name)
        print(cat_stats.to_string(index=False))
        category_results[name] = cat_stats

    # 对比分析（如果有多个模型）
    if len(args.names) >= 2:
        print("\n" + "=" * 80)
        print(f"模型对比分析: {args.names[0]} vs {args.names[1]}")
        print("=" * 80)

        df1 = all_dfs[args.names[0]]
        df2 = all_dfs[args.names[1]]

        better_in_1, better_in_2, merged = find_diff_samples(
            df1, df2, args.names[0], args.names[1], threshold=0.15
        )

        print(f"\n{args.names[0]} 表现更好的样本 ({len(better_in_1)} 个):")
        print("-" * 60)
        if len(better_in_1) > 0:
            # 统计这些样本的问题类别
            cat_counter = Counter()
            for _, row in better_in_1.iterrows():
                for cat in row[f'categories_{args.names[0]}']:
                    cat_counter[cat] += 1
            print(f"问题类别分布: {dict(cat_counter)}")

            print(f"\n前{min(args.top_n, len(better_in_1))}个样本:")
            for i, (_, row) in enumerate(better_in_1.head(args.top_n).iterrows()):
                print(f"\n  [{i+1}] idx={row['idx']}")
                print(f"      问题: {row[f'question_{args.names[0]}'][:80]}...")
                print(f"      类别: {row[f'categories_{args.names[0]}']}")
                print(f"      {args.names[0]}答案: {row[f'pred_{args.names[0]}']}")
                print(f"      {args.names[1]}答案: {row[f'pred_{args.names[1]}']}")
                print(f"      标准答案: {row[f'gold_{args.names[0]}']}")
                print(f"      得分差: {row['score_diff']:.4f} ({row[f'score_{args.names[0]}']:.4f} vs {row[f'score_{args.names[1]}']:.4f})")

        print(f"\n\n{args.names[1]} 表现更好的样本 ({len(better_in_2)} 个):")
        print("-" * 60)
        if len(better_in_2) > 0:
            cat_counter = Counter()
            for _, row in better_in_2.iterrows():
                for cat in row[f'categories_{args.names[1]}']:
                    cat_counter[cat] += 1
            print(f"问题类别分布: {dict(cat_counter)}")

            print(f"\n前{min(args.top_n, len(better_in_2))}个样本:")
            for i, (_, row) in enumerate(better_in_2.head(args.top_n).iterrows()):
                print(f"\n  [{i+1}] idx={row['idx']}")
                print(f"      问题: {row[f'question_{args.names[0]}'][:80]}...")
                print(f"      类别: {row[f'categories_{args.names[1]}']}")
                print(f"      {args.names[0]}答案: {row[f'pred_{args.names[0]}']}")
                print(f"      {args.names[1]}答案: {row[f'pred_{args.names[1]}']}")
                print(f"      标准答案: {row[f'gold_{args.names[0]}']}")
                print(f"      得分差: {row['score_diff']:.4f} ({row[f'score_{args.names[0]}']:.4f} vs {row[f'score_{args.names[1]}']:.4f})")

    # 保存详细结果
    print("\n" + "=" * 80)
    print("保存分析结果...")
    print("=" * 80)

    # 保存各模型得分详情
    for name, df in all_dfs.items():
        output_file = f"{args.output}_{name}_scores.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  {output_file}")

    # 保存类别分析
    all_cat_stats = []
    for name, cat_df in category_results.items():
        cat_df['model'] = name
        all_cat_stats.append(cat_df)

    cat_output = f"{args.output}_category_stats.csv"
    pd.concat(all_cat_stats).to_csv(cat_output, index=False, encoding='utf-8-sig')
    print(f"  {cat_output}")

    # 保存对比结果
    if len(args.names) >= 2:
        diff_output = f"{args.output}_diff_{args.names[0]}_vs_{args.names[1]}.csv"
        merged.to_csv(diff_output, index=False, encoding='utf-8-sig')
        print(f"  {diff_output}")

    print("\n分析完成！")


if __name__ == "__main__":
    main()
