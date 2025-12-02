#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCKS2025 复赛评估脚本
根据官方评分标准：
- BERTScore (F1): 50%
- ROUGE-1 (F1): 25%
- 精确匹配准确率 (EM): 25%

Final Score = 0.5 × BERTScore(F1) + 0.25 × ROUGE-1(F1) + 0.25 × EM

消融实验分析：
- 基础模型 vs 微调模型
- 各Prompt模块的贡献
"""

import json
import pandas as pd
import numpy as np
import re
import argparse
from collections import Counter
from tqdm import tqdm
import os

# 尝试导入 bert_score
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("警告: bert_score 未安装，将使用替代方案")
    print("安装命令: pip install bert-score")


# ============================================
# 文本预处理
# ============================================

def normalize_text(text):
    """文本标准化"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_for_em(text):
    """用于精确匹配的文本标准化"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize_chinese(text):
    """中文分词（按字符）"""
    return list(text.replace(' ', ''))


# ============================================
# 评估指标
# ============================================

def exact_match(pred, gold):
    """精确匹配 (EM)"""
    return float(normalize_for_em(pred) == normalize_for_em(gold))


def rouge_1_f1(pred, gold):
    """ROUGE-1 F1 分数（基于字符的unigram重叠）"""
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

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bert_score_batch(preds, golds, batch_size=32):
    """批量计算 BERTScore"""
    if not BERT_SCORE_AVAILABLE:
        print("使用字符级F1作为BERTScore替代")
        scores = [rouge_1_f1(p, g) for p, g in zip(preds, golds)]
        return scores

    print("计算 BERTScore...")
    P, R, F1 = bert_score_func(
        preds, golds,
        lang="zh",
        verbose=True,
        batch_size=batch_size,
        rescale_with_baseline=True
    )
    return F1.tolist()


def compute_bert_score_single(pred, gold):
    """计算单个样本的 BERTScore"""
    if not BERT_SCORE_AVAILABLE:
        return rouge_1_f1(pred, gold)

    P, R, F1 = bert_score_func(
        [pred], [gold],
        lang="zh",
        verbose=False,
        rescale_with_baseline=True
    )
    return F1[0].item()


# ============================================
# 综合评估
# ============================================

def evaluate_single(pred, gold, bert_f1=None):
    """评估单个样本"""
    em = exact_match(pred, gold)
    rouge1 = rouge_1_f1(pred, gold)

    if bert_f1 is None:
        bert_f1 = compute_bert_score_single(pred, gold)

    final_score = 0.5 * bert_f1 + 0.25 * rouge1 + 0.25 * em

    return {
        'exact_match': em,
        'rouge1_f1': rouge1,
        'bert_score_f1': bert_f1,
        'final_score': final_score
    }


def evaluate_results(results_file, gold_file=None, pred_key='style_answer', gold_key='answer'):
    """评估结果文件"""

    predictions = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    print(f"加载预测结果: {len(predictions)} 条")

    if gold_file:
        df_gold = pd.read_json(gold_file, lines=True)
    else:
        df_gold = pd.read_json('/usr/yuque/guo/pdf_processer/patent_b/train/train.jsonl', lines=True)

    print(f"加载标准答案: {len(df_gold)} 条")

    all_preds = []
    all_golds = []
    all_indices = []

    for pred_item in predictions:
        idx = int(pred_item['idx'])

        if idx >= len(df_gold):
            print(f"警告: 索引 {idx} 超出范围，跳过")
            continue

        pred_answer = str(pred_item.get(pred_key, pred_item.get('answer', '')))
        gold_answer = str(df_gold.loc[idx, gold_key])

        all_preds.append(pred_answer)
        all_golds.append(gold_answer)
        all_indices.append(idx)

    bert_scores = compute_bert_score_batch(all_preds, all_golds)

    all_scores = []
    for i, (pred, gold, idx) in enumerate(zip(all_preds, all_golds, all_indices)):
        em = exact_match(pred, gold)
        rouge1 = rouge_1_f1(pred, gold)
        bert_f1 = bert_scores[i]
        final_score = 0.5 * bert_f1 + 0.25 * rouge1 + 0.25 * em

        all_scores.append({
            'idx': idx,
            'pred': pred,
            'gold': gold,
            'exact_match': em,
            'rouge1_f1': rouge1,
            'bert_score_f1': bert_f1,
            'final_score': final_score
        })

    df_scores = pd.DataFrame(all_scores)

    summary = {
        'total_samples': len(df_scores),
        'exact_match': df_scores['exact_match'].mean(),
        'rouge1_f1': df_scores['rouge1_f1'].mean(),
        'bert_score_f1': df_scores['bert_score_f1'].mean(),
        'final_score': df_scores['final_score'].mean()
    }

    return summary, df_scores


def print_summary(summary, exp_name=""):
    """打印评估摘要"""
    print("\n" + "=" * 60)
    if exp_name:
        print(f"评估结果: {exp_name}")
    else:
        print("评估结果")
    print("=" * 60)
    print(f"样本数量: {summary['total_samples']}")
    print("-" * 60)
    print(f"BERTScore (F1):    {summary['bert_score_f1']*100:.2f}%  (权重: 50%)")
    print(f"ROUGE-1 (F1):      {summary['rouge1_f1']*100:.2f}%  (权重: 25%)")
    print(f"精确匹配 (EM):     {summary['exact_match']*100:.2f}%  (权重: 25%)")
    print("-" * 60)
    print(f"最终得分:          {summary['final_score']*100:.2f}%")
    print("=" * 60)
    print(f"\n公式: Final = 0.5×BERTScore + 0.25×ROUGE-1 + 0.25×EM")
    print(f"计算: {summary['final_score']*100:.2f} = 0.5×{summary['bert_score_f1']*100:.2f} + 0.25×{summary['rouge1_f1']*100:.2f} + 0.25×{summary['exact_match']*100:.2f}")


# ============================================
# 消融实验分析
# ============================================

class AblationAnalyzer:
    """消融实验分析器"""

    # 实验配置定义
    EXPERIMENTS = {
        # ==========================================
        # 模型对比实验
        # ==========================================
        'base_minimal': {
            'description': '原始模型 + 最简Prompt',
            'model': 'base',
            'prompt_modules': []
        },
        'base_full': {
            'description': '原始模型 + 完整Prompt',
            'model': 'base',
            'prompt_modules': ['question_classify', 'image_retrieval', 'answer_style', 'answer_extract', 'position_hint']
        },
        'minimal': {
            'description': '微调模型 + 最简Prompt',
            'model': 'finetuned',
            'prompt_modules': []
        },
        'full': {
            'description': '微调模型 + 完整Prompt',
            'model': 'finetuned',
            'prompt_modules': ['question_classify', 'image_retrieval', 'answer_style', 'answer_extract', 'position_hint']
        },

        # ==========================================
        # Prompt模块消融实验（基于微调模型）
        # ==========================================
        'no_classify': {
            'description': '关闭问题分类',
            'model': 'finetuned',
            'prompt_modules': ['image_retrieval', 'answer_style', 'answer_extract', 'position_hint'],
            'removed': 'question_classify'
        },
        'no_retrieval': {
            'description': '关闭图片检索',
            'model': 'finetuned',
            'prompt_modules': ['question_classify', 'answer_style', 'answer_extract', 'position_hint'],
            'removed': 'image_retrieval'
        },
        'no_style': {
            'description': '关闭答案风格匹配',
            'model': 'finetuned',
            'prompt_modules': ['question_classify', 'image_retrieval', 'answer_extract', 'position_hint'],
            'removed': 'answer_style'
        },
        'no_extract': {
            'description': '关闭答案提炼',
            'model': 'finetuned',
            'prompt_modules': ['question_classify', 'image_retrieval', 'answer_style', 'position_hint'],
            'removed': 'answer_extract'
        },
        'no_position': {
            'description': '关闭位置特殊提示',
            'model': 'finetuned',
            'prompt_modules': ['question_classify', 'image_retrieval', 'answer_style', 'answer_extract'],
            'removed': 'position_hint'
        },
    }

    PROMPT_MODULES = {
        'question_classify': '问题分类',
        'image_retrieval': '图片检索',
        'answer_style': '答案风格匹配',
        'answer_extract': '答案提炼',
        'position_hint': '位置特殊提示'
    }

    def __init__(self, gold_file=None, pred_key='style_answer'):
        self.gold_file = gold_file
        self.pred_key = pred_key
        self.results = {}

    def load_experiment(self, exp_name, results_file):
        """加载单个实验结果"""
        if not os.path.exists(results_file):
            print(f"警告: 文件不存在 {results_file}")
            return None

        summary, df_scores = evaluate_results(results_file, self.gold_file, self.pred_key)
        self.results[exp_name] = {
            'summary': summary,
            'df_scores': df_scores,
            'file': results_file
        }
        return summary

    def load_all_experiments(self, result_dir='.'):
        """自动加载目录下所有消融实验结果"""
        for exp_name in self.EXPERIMENTS.keys():
            # 尝试不同的文件名格式
            possible_files = [
                os.path.join(result_dir, f'ablation_{exp_name}_results.jsonl'),
                os.path.join(result_dir, f'{exp_name}_results.jsonl'),
                os.path.join(result_dir, f'{exp_name}.jsonl'),
            ]

            for f in possible_files:
                if os.path.exists(f):
                    print(f"加载实验 [{exp_name}]: {f}")
                    self.load_experiment(exp_name, f)
                    break

    def analyze_model_contribution(self):
        """分析模型微调的贡献"""
        print("\n" + "=" * 80)
        print("模型微调贡献分析")
        print("=" * 80)

        contributions = {}

        # 1. 微调对最简Prompt的提升
        if 'base_minimal' in self.results and 'minimal' in self.results:
            base_score = self.results['base_minimal']['summary']['final_score']
            ft_score = self.results['minimal']['summary']['final_score']
            diff = (ft_score - base_score) * 100
            contributions['微调提升(最简Prompt)'] = diff

            print(f"\n1. 微调对最简Prompt的提升:")
            print(f"   原始模型 + 最简Prompt: {base_score*100:.2f}%")
            print(f"   微调模型 + 最简Prompt: {ft_score*100:.2f}%")
            print(f"   微调贡献: {'+' if diff > 0 else ''}{diff:.2f}%")

        # 2. 微调对完整Prompt的提升
        if 'base_full' in self.results and 'full' in self.results:
            base_score = self.results['base_full']['summary']['final_score']
            ft_score = self.results['full']['summary']['final_score']
            diff = (ft_score - base_score) * 100
            contributions['微调提升(完整Prompt)'] = diff

            print(f"\n2. 微调对完整Prompt的提升:")
            print(f"   原始模型 + 完整Prompt: {base_score*100:.2f}%")
            print(f"   微调模型 + 完整Prompt: {ft_score*100:.2f}%")
            print(f"   微调贡献: {'+' if diff > 0 else ''}{diff:.2f}%")

        # 3. 计算纯模型能力提升（排除Prompt影响）
        if all(k in self.results for k in ['base_minimal', 'minimal']):
            pure_model_gain = (self.results['minimal']['summary']['final_score'] -
                              self.results['base_minimal']['summary']['final_score']) * 100
            contributions['纯模型能力提升'] = pure_model_gain

            print(f"\n3. 纯模型能力提升（最简Prompt，排除Prompt影响）:")
            print(f"   提升: {'+' if pure_model_gain > 0 else ''}{pure_model_gain:.2f}%")

        return contributions

    def analyze_prompt_contribution(self):
        """分析Prompt各模块的贡献"""
        print("\n" + "=" * 80)
        print("Prompt模块贡献分析")
        print("=" * 80)

        contributions = {}

        # 基于微调模型的完整Prompt为基线
        if 'full' not in self.results:
            print("警告: 缺少 'full' 实验结果作为基线")
            return contributions

        baseline = self.results['full']['summary']['final_score']
        print(f"\n基线 (微调模型 + 完整Prompt): {baseline*100:.2f}%")
        print("-" * 60)

        # 1. 完整Prompt相对于最简Prompt的总提升
        if 'minimal' in self.results:
            minimal_score = self.results['minimal']['summary']['final_score']
            total_prompt_gain = (baseline - minimal_score) * 100
            contributions['Prompt总贡献'] = total_prompt_gain

            print(f"\nPrompt总贡献 (完整 - 最简):")
            print(f"  微调模型 + 最简Prompt: {minimal_score*100:.2f}%")
            print(f"  微调模型 + 完整Prompt: {baseline*100:.2f}%")
            print(f"  总贡献: {'+' if total_prompt_gain > 0 else ''}{total_prompt_gain:.2f}%")

        # 2. 各模块的边际贡献
        print(f"\n各模块边际贡献 (关闭该模块后的效果损失):")
        print("-" * 60)

        module_contributions = []
        for exp_name, exp_config in self.EXPERIMENTS.items():
            if 'removed' in exp_config and exp_name in self.results:
                removed_module = exp_config['removed']
                module_name = self.PROMPT_MODULES.get(removed_module, removed_module)
                exp_score = self.results[exp_name]['summary']['final_score']
                contribution = (baseline - exp_score) * 100

                module_contributions.append({
                    'module': module_name,
                    'exp_name': exp_name,
                    'contribution': contribution,
                    'score': exp_score
                })
                contributions[module_name] = contribution

        # 按贡献排序
        module_contributions.sort(key=lambda x: x['contribution'], reverse=True)

        for mc in module_contributions:
            sign = '+' if mc['contribution'] > 0 else ''
            impact = '正向贡献' if mc['contribution'] > 0 else '负向影响'
            print(f"  {mc['module']:<15}: {sign}{mc['contribution']:.2f}% ({impact})")
            print(f"      关闭后得分: {mc['score']*100:.2f}%")

        return contributions

    def analyze_combined_contribution(self):
        """综合分析：模型 + Prompt的贡献"""
        print("\n" + "=" * 80)
        print("综合贡献分析")
        print("=" * 80)

        results_table = []

        # 收集所有实验结果
        for exp_name, exp_config in self.EXPERIMENTS.items():
            if exp_name in self.results:
                summary = self.results[exp_name]['summary']
                results_table.append({
                    'experiment': exp_name,
                    'description': exp_config['description'],
                    'model': exp_config['model'],
                    'bert_score': summary['bert_score_f1'],
                    'rouge1': summary['rouge1_f1'],
                    'em': summary['exact_match'],
                    'final_score': summary['final_score']
                })

        if not results_table:
            print("没有可用的实验结果")
            return

        # 按final_score排序
        results_table.sort(key=lambda x: x['final_score'], reverse=True)

        # 打印表格
        print(f"\n{'实验':<15}{'描述':<25}{'BERTScore':<12}{'ROUGE-1':<12}{'EM':<12}{'Final':<12}")
        print("-" * 90)

        for r in results_table:
            print(f"{r['experiment']:<15}{r['description']:<25}"
                  f"{r['bert_score']*100:.2f}%".ljust(12) +
                  f"{r['rouge1']*100:.2f}%".ljust(12) +
                  f"{r['em']*100:.2f}%".ljust(12) +
                  f"{r['final_score']*100:.2f}%".ljust(12))

        # 计算贡献分解
        print("\n" + "-" * 60)
        print("贡献分解:")

        if all(k in self.results for k in ['base_minimal', 'base_full', 'minimal', 'full']):
            base_min = self.results['base_minimal']['summary']['final_score']
            base_full = self.results['base_full']['summary']['final_score']
            ft_min = self.results['minimal']['summary']['final_score']
            ft_full = self.results['full']['summary']['final_score']

            # 各项贡献
            prompt_on_base = (base_full - base_min) * 100  # Prompt在原始模型上的贡献
            prompt_on_ft = (ft_full - ft_min) * 100        # Prompt在微调模型上的贡献
            ft_on_min = (ft_min - base_min) * 100          # 微调在最简Prompt上的贡献
            ft_on_full = (ft_full - base_full) * 100       # 微调在完整Prompt上的贡献

            print(f"\n  基础模型得分 (最简Prompt):     {base_min*100:.2f}%")
            print(f"  + Prompt贡献 (在原始模型上):   {'+' if prompt_on_base > 0 else ''}{prompt_on_base:.2f}%")
            print(f"  = 原始模型 + 完整Prompt:       {base_full*100:.2f}%")
            print()
            print(f"  基础模型得分 (最简Prompt):     {base_min*100:.2f}%")
            print(f"  + 微调贡献 (在最简Prompt上):   {'+' if ft_on_min > 0 else ''}{ft_on_min:.2f}%")
            print(f"  = 微调模型 + 最简Prompt:       {ft_min*100:.2f}%")
            print()
            print(f"  微调模型 (最简Prompt):         {ft_min*100:.2f}%")
            print(f"  + Prompt贡献 (在微调模型上):   {'+' if prompt_on_ft > 0 else ''}{prompt_on_ft:.2f}%")
            print(f"  = 微调模型 + 完整Prompt:       {ft_full*100:.2f}%")

            print("\n" + "-" * 60)
            print("总结:")
            total_gain = (ft_full - base_min) * 100
            print(f"  总提升: {base_min*100:.2f}% → {ft_full*100:.2f}% ({'+' if total_gain > 0 else ''}{total_gain:.2f}%)")

            # 估算各因素占比
            if total_gain > 0:
                # 使用Shapley值的简化版本估算
                ft_contribution = (ft_on_min + ft_on_full) / 2
                prompt_contribution = (prompt_on_base + prompt_on_ft) / 2

                print(f"\n  估算贡献占比:")
                print(f"    - 模型微调贡献: ~{ft_contribution:.2f}% ({ft_contribution/total_gain*100:.1f}%)")
                print(f"    - Prompt贡献:   ~{prompt_contribution:.2f}% ({prompt_contribution/total_gain*100:.1f}%)")

    def generate_report(self):
        """生成完整的消融实验报告"""
        print("\n" + "=" * 80)
        print("CCKS2025 消融实验报告")
        print("=" * 80)

        print(f"\n已加载实验: {list(self.results.keys())}")
        print(f"缺失实验: {[k for k in self.EXPERIMENTS.keys() if k not in self.results]}")

        # 1. 模型贡献分析
        model_contrib = self.analyze_model_contribution()

        # 2. Prompt贡献分析
        prompt_contrib = self.analyze_prompt_contribution()

        # 3. 综合分析
        self.analyze_combined_contribution()

        # 4. 生成总结
        print("\n" + "=" * 80)
        print("关键发现")
        print("=" * 80)

        if 'full' in self.results and 'base_minimal' in self.results:
            best = self.results['full']['summary']['final_score']
            worst = self.results['base_minimal']['summary']['final_score']
            improvement = (best - worst) * 100

            print(f"\n1. 最佳配置: 微调模型 + 完整Prompt ({best*100:.2f}%)")
            print(f"2. 基线配置: 原始模型 + 最简Prompt ({worst*100:.2f}%)")
            print(f"3. 总体提升: {improvement:.2f}%")

        # 找出贡献最大的模块
        if prompt_contrib:
            sorted_modules = sorted(prompt_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            if sorted_modules:
                top_module = sorted_modules[0]
                print(f"\n4. 贡献最大的Prompt模块: {top_module[0]} ({'+' if top_module[1] > 0 else ''}{top_module[1]:.2f}%)")

        return {
            'model_contribution': model_contrib,
            'prompt_contribution': prompt_contrib
        }


def compare_experiments(result_files, gold_file=None, pred_key='style_answer'):
    """比较多个实验结果"""
    all_summaries = {}

    for result_file in result_files:
        exp_name = result_file.replace('ablation_', '').replace('_results.jsonl', '').replace('.jsonl', '')
        print(f"\n处理实验: {exp_name}")
        summary, _ = evaluate_results(result_file, gold_file, pred_key)
        all_summaries[exp_name] = summary
        print_summary(summary, exp_name)

    # 输出对比表格
    print("\n" + "=" * 80)
    print("实验对比汇总")
    print("=" * 80)

    header = f"{'实验名':<20}{'BERTScore':<15}{'ROUGE-1':<15}{'EM':<15}{'Final Score':<15}"
    print(header)
    print("-" * 80)

    sorted_exps = sorted(all_summaries.items(), key=lambda x: x[1]['final_score'], reverse=True)

    for exp_name, summary in sorted_exps:
        row = f"{exp_name:<20}"
        row += f"{summary['bert_score_f1']*100:.2f}%".ljust(15)
        row += f"{summary['rouge1_f1']*100:.2f}%".ljust(15)
        row += f"{summary['exact_match']*100:.2f}%".ljust(15)
        row += f"{summary['final_score']*100:.2f}%".ljust(15)
        print(row)

    print("=" * 80)

    if 'full' in all_summaries:
        print("\n模块贡献分析 (相对于 full 基线):")
        print("-" * 60)
        baseline = all_summaries['full']['final_score']

        for exp_name, summary in all_summaries.items():
            if exp_name != 'full':
                diff = (baseline - summary['final_score']) * 100
                if diff > 0:
                    print(f"  {exp_name}: -{diff:.2f}% (关闭该模块降低效果)")
                else:
                    print(f"  {exp_name}: +{-diff:.2f}% (关闭该模块提升效果)")

    return all_summaries


# ============================================
# 错误分析
# ============================================

def error_analysis(df_scores, threshold=0.5, top_n=10):
    """错误分析：找出低分样本"""
    low_score_samples = df_scores[df_scores['final_score'] < threshold].sort_values('final_score')

    print(f"\n" + "=" * 60)
    print(f"错误分析 (Final Score < {threshold*100:.0f}%)")
    print("=" * 60)
    print(f"低分样本数量: {len(low_score_samples)} / {len(df_scores)} ({len(low_score_samples)/len(df_scores)*100:.1f}%)")

    if len(low_score_samples) > 0:
        print(f"\n最差的 {min(top_n, len(low_score_samples))} 个样本:")
        print("-" * 60)

        for _, row in low_score_samples.head(top_n).iterrows():
            print(f"\n索引: {row['idx']}")
            print(f"预测: {row['pred'][:100]}{'...' if len(str(row['pred'])) > 100 else ''}")
            print(f"标准: {row['gold'][:100]}{'...' if len(str(row['gold'])) > 100 else ''}")
            print(f"得分: BERTScore={row['bert_score_f1']*100:.1f}%, ROUGE-1={row['rouge1_f1']*100:.1f}%, EM={row['exact_match']*100:.0f}%, Final={row['final_score']*100:.1f}%")

    print("\n" + "-" * 60)
    print("得分分布:")
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    df_scores_copy = df_scores.copy()
    df_scores_copy['score_bin'] = pd.cut(df_scores_copy['final_score'], bins=bins, labels=labels)
    distribution = df_scores_copy['score_bin'].value_counts().sort_index()
    for label, count in distribution.items():
        pct = count / len(df_scores) * 100
        print(f"  {label}: {count} 个 ({pct:.1f}%)")


def high_score_analysis(df_scores, threshold=0.8, top_n=5):
    """高分样本分析"""
    high_score_samples = df_scores[df_scores['final_score'] >= threshold].sort_values('final_score', ascending=False)

    print(f"\n" + "=" * 60)
    print(f"高分样本分析 (Final Score >= {threshold*100:.0f}%)")
    print("=" * 60)
    print(f"高分样本数量: {len(high_score_samples)} / {len(df_scores)} ({len(high_score_samples)/len(df_scores)*100:.1f}%)")

    if len(high_score_samples) > 0:
        print(f"\n最好的 {min(top_n, len(high_score_samples))} 个样本:")
        print("-" * 60)

        for _, row in high_score_samples.head(top_n).iterrows():
            print(f"\n索引: {row['idx']}")
            print(f"预测: {row['pred'][:100]}{'...' if len(str(row['pred'])) > 100 else ''}")
            print(f"标准: {row['gold'][:100]}{'...' if len(str(row['gold'])) > 100 else ''}")
            print(f"得分: Final={row['final_score']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCKS2025 复赛评估脚本')
    parser.add_argument('--results', type=str, default=None,
                        help='预测结果文件（jsonl格式）')
    parser.add_argument('--gold', type=str, default=None,
                        help='标准答案文件（jsonl格式）')
    parser.add_argument('--pred_key', type=str, default='style_answer',
                        help='预测答案的字段名 (默认: style_answer)')
    parser.add_argument('--gold_key', type=str, default='answer',
                        help='标准答案的字段名 (默认: answer)')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='多个结果文件进行对比')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验分析')
    parser.add_argument('--ablation_dir', type=str, default='.',
                        help='消融实验结果目录')
    parser.add_argument('--error_analysis', action='store_true',
                        help='是否进行错误分析')
    parser.add_argument('--save', action='store_true',
                        help='是否保存详细评估结果')

    args = parser.parse_args()

    if args.ablation:
        # 运行消融实验分析
        analyzer = AblationAnalyzer(gold_file=args.gold, pred_key=args.pred_key)
        analyzer.load_all_experiments(args.ablation_dir)
        analyzer.generate_report()
    elif args.compare:
        # 比较多个实验
        compare_experiments(args.compare, args.gold, args.pred_key)
    elif args.results:
        # 评估单个结果
        summary, df_scores = evaluate_results(
            args.results, args.gold, args.pred_key, args.gold_key
        )
        print_summary(summary)

        if args.error_analysis:
            error_analysis(df_scores)
            high_score_analysis(df_scores)

        if args.save:
            output_file = args.results.replace('.jsonl', '_eval.csv')
            df_scores.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n详细评估结果已保存到: {output_file}")

            summary_file = args.results.replace('.jsonl', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"汇总结果已保存到: {summary_file}")
    else:
        parser.print_help()
