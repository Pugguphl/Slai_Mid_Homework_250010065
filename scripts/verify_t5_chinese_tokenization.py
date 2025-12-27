#!/usr/bin/env python3
"""
验证T5在中文翻译任务上的分词问题

验证要点：
1. T5预训练语料以英语为主，词表对中文覆盖不足
2. SentencePiece词表对中文字符/子词覆盖不友好
3. 中文被切得很碎或大量<unk>/低质量子词
4. 编码困难导致翻译质量下降
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False


def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_chinese_chars(text: str) -> Set[str]:
    """提取中文字符"""
    return {char for char in text if '\u4e00' <= char <= '\u9fff'}


def analyze_t5_tokenizer_coverage(tokenizer, texts: List[str]) -> Dict:
    """
    分析T5 tokenizer的中文覆盖情况
    
    验证：
    1. <unk> token数量和比例
    2. 中文字符被切分的程度（单个中文字符是否需要多个token）
    3. 中文句子的token化长度
    """
    results = {
        'unk_count': 0,
        'total_tokens': 0,
        'total_chinese_chars': 0,
        'chinese_char_token_ratio': [],  # 每个中文字符需要的token数
        'sentence_token_counts': [],
        'single_char_tokens': 0,  # 单个中文字符被切分成多个token的情况
        'multi_char_tokens': 0,  # 多个中文字符组成一个token的情况
        'token_fragments': [],  # 记录token碎片化程度
        'examples': [],  # 记录典型示例
    }
    
    unk_token_id = tokenizer.unk_token_id
    
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        
        results['total_tokens'] += len(tokens)
        results['sentence_token_counts'].append(len(tokens))
        
        chinese_chars = get_chinese_chars(text)
        results['total_chinese_chars'] += len(chinese_chars)
        
        for token_id, token_str in zip(tokens, decoded_tokens):
            if token_id == unk_token_id:
                results['unk_count'] += 1
            
            token_chinese_chars = get_chinese_chars(token_str)
            if token_chinese_chars:
                if len(token_chinese_chars) == 1:
                    results['single_char_tokens'] += 1
                else:
                    results['multi_char_tokens'] += 1
                
                for char in token_chinese_chars:
                    char_tokens = tokenizer.encode(char, add_special_tokens=False)
                    results['chinese_char_token_ratio'].append(len(char_tokens))
        
        if i < 10:
            results['examples'].append({
                'text': text,
                'tokens': decoded_tokens,
                'token_count': len(tokens),
                'unk_count': sum(1 for t in tokens if t == unk_token_id),
            })
    
    results['unk_rate'] = results['unk_count'] / results['total_tokens'] if results['total_tokens'] > 0 else 0
    results['avg_tokens_per_sentence'] = np.mean(results['sentence_token_counts'])
    results['avg_tokens_per_chinese_char'] = np.mean(results['chinese_char_token_ratio']) if results['chinese_char_token_ratio'] else 0
    results['fragmentation_rate'] = results['single_char_tokens'] / (results['single_char_tokens'] + results['multi_char_tokens']) if (results['single_char_tokens'] + results['multi_char_tokens']) > 0 else 0
    
    return results


def analyze_subword_quality(tokenizer, texts: List[str]) -> Dict:
    """
    分析子词质量
    
    验证：
    1. 子词的语义完整性（是否经常出现无意义的碎片）
    2. 中文子词的长度分布
    3. 高频子词统计
    """
    subword_counter = Counter()
    chinese_subword_counter = Counter()
    subword_lengths = defaultdict(int)
    chinese_subword_lengths = defaultdict(int)
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        
        for token_str in decoded_tokens:
            subword_counter[token_str] += 1
            subword_lengths[len(token_str)] += 1
            
            chinese_chars = get_chinese_chars(token_str)
            if chinese_chars:
                chinese_subword_counter[token_str] += 1
                chinese_subword_lengths[len(token_str)] += 1
    
    return {
        'total_unique_subwords': len(subword_counter),
        'chinese_subwords': len(chinese_subword_counter),
        'top_20_subwords': subword_counter.most_common(20),
        'top_20_chinese_subwords': chinese_subword_counter.most_common(20),
        'subword_length_distribution': dict(subword_lengths),
        'chinese_subword_length_distribution': dict(chinese_subword_lengths),
    }


def analyze_encoding_efficiency(t5_tokenizer, spm_tokenizer, texts: List[str]) -> Dict:
    """
    对比T5和SentencePiece的编码效率
    
    验证：
    1. 同样的中文句子，T5需要更多的token
    2. T5的token碎片化程度更高
    """
    t5_token_counts = []
    spm_token_counts = []
    t5_unk_counts = []
    spm_unk_counts = []
    
    for text in texts:
        t5_tokens = t5_tokenizer.encode(text, add_special_tokens=False)
        spm_tokens = spm_tokenizer.encode(text, out_type=int)
        
        t5_token_counts.append(len(t5_tokens))
        spm_token_counts.append(len(spm_tokens))
        
        t5_unk_counts.append(sum(1 for t in t5_tokens if t == t5_tokenizer.unk_token_id))
        spm_unk_counts.append(sum(1 for t in spm_tokens if spm_tokenizer.id_to_piece(t) == '<unk>'))
    
    return {
        'avg_t5_tokens': np.mean(t5_token_counts),
        'avg_spm_tokens': np.mean(spm_token_counts),
        'token_ratio': np.mean(t5_token_counts) / np.mean(spm_token_counts) if np.mean(spm_token_counts) > 0 else 0,
        'avg_t5_unk': np.mean(t5_unk_counts),
        'avg_spm_unk': np.mean(spm_unk_counts),
        't5_token_counts': t5_token_counts,
        'spm_token_counts': spm_token_counts,
    }


def print_t5_analysis_report(t5_coverage: Dict, t5_subword: Dict, efficiency: Dict):
    """打印T5分析报告"""
    print("=" * 80)
    print("T5中文分词问题验证报告")
    print("=" * 80)
    print()
    
    print("【问题1: <unk> Token问题】")
    print("-" * 80)
    print(f"总Token数: {t5_coverage['total_tokens']}")
    print(f"<unk>数量: {t5_coverage['unk_count']}")
    print(f"<unk>比例: {t5_coverage['unk_rate']*100:.4f}%")
    if t5_coverage['unk_rate'] > 0:
        print("⚠️  警告: 存在<unk> token，说明词表覆盖不足！")
    print()
    
    print("【问题2: 中文碎片化问题】")
    print("-" * 80)
    print(f"平均每句Token数: {t5_coverage['avg_tokens_per_sentence']:.2f}")
    print(f"平均每个中文字符需要的Token数: {t5_coverage['avg_tokens_per_chinese_char']:.2f}")
    print(f"碎片化率（单个字符被切分）: {t5_coverage['fragmentation_rate']*100:.2f}%")
    if t5_coverage['avg_tokens_per_chinese_char'] > 1.2:
        print("⚠️  警告: 中文字符被过度切分，碎片化严重！")
    print()
    
    print("【问题3: 子词质量分析】")
    print("-" * 80)
    print(f"总唯一子词数: {t5_subword['total_unique_subwords']}")
    print(f"中文相关子词数: {t5_subword['chinese_subwords']}")
    print(f"中文子词占比: {t5_subword['chinese_subwords']/t5_subword['total_unique_subwords']*100:.2f}%")
    if t5_subword['chinese_subwords'] / t5_subword['total_unique_subwords'] < 0.1:
        print("⚠️  警告: 词表中中文子词占比过低，说明预训练语料以英语为主！")
    print()
    
    print("【Top 20 高频子词】")
    print("-" * 80)
    for i, (token, count) in enumerate(t5_subword['top_20_subwords'], 1):
        chinese_mark = " [中文]" if get_chinese_chars(token) else ""
        print(f"{i:2d}. '{token}' (出现{count}次){chinese_mark}")
    print()
    
    print("【Top 20 中文子词】")
    print("-" * 80)
    for i, (token, count) in enumerate(t5_subword['top_20_chinese_subwords'], 1):
        print(f"{i:2d}. '{token}' (出现{count}次)")
    print()
    
    print("【问题4: 编码效率对比（T5 vs SentencePiece）】")
    print("-" * 80)
    print(f"T5平均Token数: {efficiency['avg_t5_tokens']:.2f}")
    print(f"SentencePiece平均Token数: {efficiency['avg_spm_tokens']:.2f}")
    print(f"T5/SentencePiece比率: {efficiency['token_ratio']:.2f}x")
    print(f"T5平均<unk>数: {efficiency['avg_t5_unk']:.4f}")
    print(f"SentencePiece平均<unk>数: {efficiency['avg_spm_unk']:.4f}")
    if efficiency['token_ratio'] > 1.5:
        print("⚠️  警告: T5编码效率明显低于SentencePiece，需要更多token！")
    print()
    
    print("【典型示例】")
    print("-" * 80)
    for i, example in enumerate(t5_coverage['examples'][:5], 1):
        print(f"示例 {i}: {example['text']}")
        print(f"  Token数: {example['token_count']}, <unk>数: {example['unk_count']}")
        print(f"  Tokens: {example['tokens']}")
        print()


def visualize_t5_problems(t5_coverage: Dict, t5_subword: Dict, efficiency: Dict, output_dir: Path):
    """生成可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('T5中文分词问题验证', fontsize=18, fontweight='bold')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['<unk>比例'], [t5_coverage['unk_rate']*100], color=colors[0])
    ax1.set_ylabel('比例 (%)', fontsize=11)
    ax1.set_title('<unk> Token比例', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0, t5_coverage['unk_rate']*100 + 0.5, f"{t5_coverage['unk_rate']*100:.4f}%", 
             ha='center', fontsize=10, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['碎片化率'], [t5_coverage['fragmentation_rate']*100], color=colors[1])
    ax2.set_ylabel('比例 (%)', fontsize=11)
    ax2.set_title('中文碎片化率', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, t5_coverage['fragmentation_rate']*100 + 1, 
             f"{t5_coverage['fragmentation_rate']*100:.2f}%", 
             ha='center', fontsize=10, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    chinese_ratio = t5_subword['chinese_subwords'] / t5_subword['total_unique_subwords'] * 100
    ax3.bar(['中文子词占比'], [chinese_ratio], color=colors[2])
    ax3.set_ylabel('比例 (%)', fontsize=11)
    ax3.set_title('词表中中文子词占比', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(0, chinese_ratio + 1, f"{chinese_ratio:.2f}%", 
             ha='center', fontsize=10, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, :])
    lengths = list(t5_subword['subword_length_distribution'].keys())
    counts = list(t5_subword['subword_length_distribution'].values())
    ax4.bar(lengths, counts, color=colors[3], alpha=0.7)
    ax4.set_xlabel('子词长度（字符数）', fontsize=11)
    ax4.set_ylabel('频次', fontsize=11)
    ax4.set_title('子词长度分布', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(['T5', 'SentencePiece'], 
            [efficiency['avg_t5_tokens'], efficiency['avg_spm_tokens']], 
            color=[colors[0], colors[1]])
    ax5.set_ylabel('平均Token数', fontsize=11)
    ax5.set_title('编码效率对比', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate([efficiency['avg_t5_tokens'], efficiency['avg_spm_tokens']]):
        ax5.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.bar(['T5', 'SentencePiece'], 
            [efficiency['avg_t5_unk'], efficiency['avg_spm_unk']], 
            color=[colors[0], colors[1]])
    ax6.set_ylabel('平均<unk>数', fontsize=11)
    ax6.set_title('<unk> Token对比', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate([efficiency['avg_t5_unk'], efficiency['avg_spm_unk']]):
        ax6.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
    
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.bar(['Token比率'], [efficiency['token_ratio']], color=colors[2])
    ax7.set_ylabel('T5/SentencePiece', fontsize=11)
    ax7.set_title('Token数比率', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    ax7.text(0, efficiency['token_ratio'] + 0.05, f"{efficiency['token_ratio']:.2f}x", 
             ha='center', fontsize=10, fontweight='bold')
    
    output_path = output_dir / 't5_chinese_tokenization_problems.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存到: {output_path}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(t5_coverage['chinese_char_token_ratio'], bins=30, 
            color=colors[0], alpha=0.7, edgecolor='black')
    ax.axvline(t5_coverage['avg_tokens_per_chinese_char'], 
               color=colors[1], linestyle='--', linewidth=2, 
               label=f'平均值: {t5_coverage["avg_tokens_per_chinese_char"]:.2f}')
    ax.set_xlabel('每个中文字符需要的Token数', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('中文字符Token化分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 't5_chinese_char_token_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 中文字符Token化分布图已保存到: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("T5中文分词问题验证")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).resolve().parents[1]
    
    data_dir = project_root / "data" / "processed"
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    
    print("加载数据...")
    train_data = load_jsonl(train_file)
    valid_data = load_jsonl(valid_file)
    
    zh_texts = [ex['zh'] for ex in train_data + valid_data]
    print(f"✓ 加载了 {len(zh_texts)} 个中文句子")
    print()
    
    print("加载T5 tokenizer...")
    t5_tokenizer = AutoTokenizer.from_pretrained(project_root / "models" / "t5-small")
    print(f"✓ T5 tokenizer加载完成 (vocab_size={t5_tokenizer.vocab_size})")
    print()
    
    print("加载SentencePiece tokenizer...")
    spm_model_path = project_root / "data" / "vocab" / "spm_zh_en.model"
    spm_tokenizer = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    print(f"✓ SentencePiece tokenizer加载完成 (vocab_size={spm_tokenizer.vocab_size()})")
    print()
    
    print("分析T5 tokenizer的中文覆盖情况...")
    t5_coverage = analyze_t5_tokenizer_coverage(t5_tokenizer, zh_texts)
    print("✓ T5覆盖分析完成")
    print()
    
    print("分析T5子词质量...")
    t5_subword = analyze_subword_quality(t5_tokenizer, zh_texts)
    print("✓ T5子词质量分析完成")
    print()
    
    print("对比T5和SentencePiece的编码效率...")
    efficiency = analyze_encoding_efficiency(t5_tokenizer, spm_tokenizer, zh_texts)
    print("✓ 编码效率对比完成")
    print()
    
    print_t5_analysis_report(t5_coverage, t5_subword, efficiency)
    
    output_dir = project_root / "experiments" / "results" / "t5_chinese_analysis"
    visualize_t5_problems(t5_coverage, t5_subword, efficiency, output_dir)
    
    results = {
        'summary': {
            'unk_rate': t5_coverage['unk_rate'],
            'fragmentation_rate': t5_coverage['fragmentation_rate'],
            'avg_tokens_per_chinese_char': t5_coverage['avg_tokens_per_chinese_char'],
            'chinese_subword_ratio': t5_subword['chinese_subwords'] / t5_subword['total_unique_subwords'],
            't5_vs_spm_token_ratio': efficiency['token_ratio'],
        },
        't5_coverage': t5_coverage,
        't5_subword': t5_subword,
        'efficiency': efficiency,
    }
    
    results_file = output_dir / "t5_chinese_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 分析结果已保存到: {results_file}")
    print()
    
    print("=" * 80)
    print("T5中文分词问题验证完成!")
    print("=" * 80)
    print()
    print("结论:")
    print("-" * 80)
    if t5_coverage['unk_rate'] > 0:
        print("1. ✓ 验证: T5存在<unk> token，词表覆盖不足")
    if t5_coverage['avg_tokens_per_chinese_char'] > 1.2:
        print("2. ✓ 验证: 中文字符被过度切分，碎片化严重")
    if t5_subword['chinese_subwords'] / t5_subword['total_unique_subwords'] < 0.1:
        print("3. ✓ 验证: 词表中中文子词占比过低，预训练语料以英语为主")
    if efficiency['token_ratio'] > 1.5:
        print("4. ✓ 验证: T5编码效率明显低于SentencePiece，需要更多token")
    print("-" * 80)


if __name__ == '__main__':
    main()
