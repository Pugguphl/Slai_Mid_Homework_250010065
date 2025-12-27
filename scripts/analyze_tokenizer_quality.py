#!/usr/bin/env python3
"""
分词质量对比分析脚本
对比T5 tokenizer和自定义SentencePiece tokenizer对中文的分词质量
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_tokenizer(
    tokenizer,
    texts: List[str],
    tokenizer_name: str,
    is_t5: bool = False
) -> Dict:
    """
    分析tokenizer的分词质量
    
    返回指标：
    - unk_count: <unk> token数量
    - unk_rate: <unk>比例
    - total_tokens: 总token数
    - avg_tokens_per_sentence: 平均每句token数
    - avg_token_length: 平均token长度（字符数）
    - max_token_length: 最大token长度
    - chinese_char_coverage: 中文字符覆盖率
    - subword_stats: 子词统计
    """
    results = {
        'tokenizer_name': tokenizer_name,
        'unk_count': 0,
        'total_tokens': 0,
        'token_lengths': [],
        'subword_pieces': [],
        'chinese_chars': set(),
        'chinese_chars_covered': set(),
        'sentence_token_counts': [],
    }
    
    unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 0
    unk_token = tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else '<unk>'
    
    for text in texts:
        if is_t5:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        else:
            tokens = tokenizer.encode(text, out_type=int)
            decoded_tokens = [tokenizer.id_to_piece(t) for t in tokens]
        
        results['total_tokens'] += len(tokens)
        results['sentence_token_counts'].append(len(tokens))
        
        for token_id, token_str in zip(tokens, decoded_tokens):
            if token_id == unk_token_id or token_str == unk_token or token_str == '<unk>':
                results['unk_count'] += 1
            
            results['token_lengths'].append(len(token_str))
            results['subword_pieces'].append(token_str)
            
            for char in token_str:
                if '\u4e00' <= char <= '\u9fff':
                    results['chinese_chars'].add(char)
                    results['chinese_chars_covered'].add(char)
    
    results['unk_rate'] = results['unk_count'] / results['total_tokens'] if results['total_tokens'] > 0 else 0
    results['avg_tokens_per_sentence'] = np.mean(results['sentence_token_counts'])
    results['avg_token_length'] = np.mean(results['token_lengths']) if results['token_lengths'] else 0
    results['max_token_length'] = max(results['token_lengths']) if results['token_lengths'] else 0
    results['chinese_char_coverage'] = len(results['chinese_chars_covered'])
    
    return results


def analyze_subword_distribution(tokenizer, texts: List[str], tokenizer_name: str, is_t5: bool = False) -> Dict:
    """
    分析子词分布
    """
    subword_counter = Counter()
    subword_lengths = defaultdict(int)
    
    for text in texts[:1000]:
        if is_t5:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        else:
            tokens = tokenizer.encode(text, out_type=int)
            decoded_tokens = [tokenizer.id_to_piece(t) for t in tokens]
        
        for token_str in decoded_tokens:
            subword_counter[token_str] += 1
            subword_lengths[len(token_str)] += 1
    
    return {
        'tokenizer_name': tokenizer_name,
        'unique_subwords': len(subword_counter),
        'top_10_subwords': subword_counter.most_common(10),
        'subword_length_distribution': dict(subword_lengths),
    }


def print_comparison(t5_results: Dict, spm_results: Dict, t5_subword: Dict, spm_subword: Dict):
    print("=" * 80)
    print("分词质量对比分析")
    print("=" * 80)
    print()
    
    print("【基础指标对比】")
    print("-" * 80)
    print(f"{'指标':<30} {'T5 Tokenizer':<20} {'SentencePiece':<20}")
    print("-" * 80)
    print(f"{'总Token数':<30} {t5_results['total_tokens']:<20} {spm_results['total_tokens']:<20}")
    print(f"{'<unk>数量':<30} {t5_results['unk_count']:<20} {spm_results['unk_count']:<20}")
    print(f"{'<unk>比例':<30} {t5_results['unk_rate']*100:.2f}%{'':<15} {spm_results['unk_rate']*100:.2f}%{'':<15}")
    print(f"{'平均每句Token数':<30} {t5_results['avg_tokens_per_sentence']:.2f}{'':<16} {spm_results['avg_tokens_per_sentence']:.2f}{'':<16}")
    print(f"{'平均Token长度':<30} {t5_results['avg_token_length']:.2f}{'':<16} {spm_results['avg_token_length']:.2f}{'':<16}")
    print(f"{'最大Token长度':<30} {t5_results['max_token_length']:<20} {spm_results['max_token_length']:<20}")
    print(f"{'中文字符覆盖数':<30} {t5_results['chinese_char_coverage']:<20} {spm_results['chinese_char_coverage']:<20}")
    print()
    
    print("【子词分布对比】")
    print("-" * 80)
    print(f"{'指标':<30} {'T5 Tokenizer':<20} {'SentencePiece':<20}")
    print("-" * 80)
    print(f"{'唯一子词数':<30} {t5_subword['unique_subwords']:<20} {spm_subword['unique_subwords']:<20}")
    print()
    
    print("【T5 Tokenizer - Top 10 子词】")
    print("-" * 80)
    for i, (token, count) in enumerate(t5_subword['top_10_subwords'], 1):
        print(f"{i}. '{token}' (出现{count}次)")
    print()
    
    print("【SentencePiece - Top 10 子词】")
    print("-" * 80)
    for i, (token, count) in enumerate(spm_subword['top_10_subwords'], 1):
        print(f"{i}. '{token}' (出现{count}次)")
    print()
    
    print("【Token长度分布对比】")
    print("-" * 80)
    print(f"{'Token长度':<15} {'T5 Tokenizer':<20} {'SentencePiece':<20}")
    print("-" * 80)
    all_lengths = sorted(set(list(t5_subword['subword_length_distribution'].keys()) + 
                            list(spm_subword['subword_length_distribution'].keys())))
    for length in all_lengths[:10]:
        t5_count = t5_subword['subword_length_distribution'].get(length, 0)
        spm_count = spm_subword['subword_length_distribution'].get(length, 0)
        print(f"{length:<15} {t5_count:<20} {spm_count:<20}")
    print()


def visualize_comparison(t5_results: Dict, spm_results: Dict, output_dir: Path):
    """生成可视化对比图"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T5 Tokenizer vs SentencePiece 分词质量对比', fontsize=16, fontweight='bold')
    
    tokenizer_names = ['T5 Tokenizer', 'SentencePiece']
    unk_rates = [t5_results['unk_rate'] * 100, spm_results['unk_rate'] * 100]
    avg_token_counts = [t5_results['avg_tokens_per_sentence'], spm_results['avg_tokens_per_sentence']]
    avg_token_lengths = [t5_results['avg_token_length'], spm_results['avg_token_length']]
    chinese_coverage = [t5_results['chinese_char_coverage'], spm_results['chinese_char_coverage']]
    
    colors = ['#ff6b6b', '#4ecdc4']
    
    axes[0, 0].bar(tokenizer_names, unk_rates, color=colors)
    axes[0, 0].set_ylabel('<unk> 比例 (%)', fontsize=12)
    axes[0, 0].set_title('<unk> Token 比例对比', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].bar(tokenizer_names, avg_token_counts, color=colors)
    axes[0, 1].set_ylabel('平均每句Token数', fontsize=12)
    axes[0, 1].set_title('句子Token化长度对比', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    axes[1, 0].bar(tokenizer_names, avg_token_lengths, color=colors)
    axes[1, 0].set_ylabel('平均Token长度 (字符)', fontsize=12)
    axes[1, 0].set_title('Token语义密度对比', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    axes[1, 1].bar(tokenizer_names, chinese_coverage, color=colors)
    axes[1, 1].set_ylabel('中文字符覆盖数', fontsize=12)
    axes[1, 1].set_title('中文分词覆盖能力对比', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'tokenizer_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存到: {output_path}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    t5_lengths = t5_results['token_lengths']
    spm_lengths = spm_results['token_lengths']
    
    ax.hist(t5_lengths, bins=50, alpha=0.6, label='T5 Tokenizer', color='#ff6b6b', density=True)
    ax.hist(spm_lengths, bins=50, alpha=0.6, label='SentencePiece', color='#4ecdc4', density=True)
    ax.set_xlabel('Token长度 (字符数)', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('Token长度分布对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'token_length_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Token长度分布图已保存到: {output_path}")
    plt.close()


def show_example_tokenization(t5_tokenizer, spm_tokenizer, texts: List[str], num_examples: int = 5):
    """展示具体句子的分词示例"""
    print("=" * 80)
    print(f"【分词示例对比】(随机选取{num_examples}个句子)")
    print("=" * 80)
    print()
    
    import random
    examples = random.sample(texts, min(num_examples, len(texts)))
    
    for i, text in enumerate(examples, 1):
        print(f"示例 {i}: {text}")
        print("-" * 80)
        
        t5_tokens = t5_tokenizer.encode(text, add_special_tokens=False)
        t5_decoded = [t5_tokenizer.decode([t]) for t in t5_tokens]
        
        spm_tokens = spm_tokenizer.encode(text, out_type=int)
        spm_decoded = [spm_tokenizer.id_to_piece(t) for t in spm_tokens]
        
        print(f"T5 Tokenizer ({len(t5_tokens)} tokens):")
        print(f"  Tokens: {t5_decoded}")
        print(f"  IDs: {t5_tokens}")
        print()
        
        print(f"SentencePiece ({len(spm_tokens)} tokens):")
        print(f"  Tokens: {spm_decoded}")
        print(f"  IDs: {spm_tokens}")
        print()
        
        t5_unk_count = sum(1 for t in t5_tokens if t == t5_tokenizer.unk_token_id)
        spm_unk_count = sum(1 for t in spm_tokens if spm_tokenizer.id_to_piece(t) == '<unk>')
        
        print(f"对比: T5有{t5_unk_count}个<unk>, SentencePiece有{spm_unk_count}个<unk>")
        print("=" * 80)
        print()


def main():
    print("=" * 80)
    print("T5 vs SentencePiece 分词质量对比分析")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).resolve().parents[2]
    
    data_dir = project_root / "data" / "processed"
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    
    print(f"加载数据...")
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
    
    print("分析T5 tokenizer...")
    t5_results = analyze_tokenizer(t5_tokenizer, zh_texts, "T5 Tokenizer", is_t5=True)
    print(f"✓ T5 tokenizer分析完成")
    print()
    
    print("分析SentencePiece tokenizer...")
    spm_results = analyze_tokenizer(spm_tokenizer, zh_texts, "SentencePiece", is_t5=False)
    print(f"✓ SentencePiece tokenizer分析完成")
    print()
    
    print("分析子词分布...")
    t5_subword = analyze_subword_distribution(t5_tokenizer, zh_texts, "T5 Tokenizer", is_t5=True)
    spm_subword = analyze_subword_distribution(spm_tokenizer, zh_texts, "SentencePiece", is_t5=False)
    print(f"✓ 子词分布分析完成")
    print()
    
    print_comparison(t5_results, spm_results, t5_subword, spm_subword)
    
    show_example_tokenization(t5_tokenizer, spm_tokenizer, zh_texts, num_examples=3)
    
    output_dir = project_root / "experiments" / "results" / "tokenizer_analysis"
    visualize_comparison(t5_results, spm_results, output_dir)
    
    results = {
        't5': {
            'unk_rate': t5_results['unk_rate'],
            'avg_tokens_per_sentence': t5_results['avg_tokens_per_sentence'],
            'avg_token_length': t5_results['avg_token_length'],
            'chinese_char_coverage': t5_results['chinese_char_coverage'],
            'unique_subwords': t5_subword['unique_subwords'],
        },
        'sentencepiece': {
            'unk_rate': spm_results['unk_rate'],
            'avg_tokens_per_sentence': spm_results['avg_tokens_per_sentence'],
            'avg_token_length': spm_results['avg_token_length'],
            'chinese_char_coverage': spm_results['chinese_char_coverage'],
            'unique_subwords': spm_subword['unique_subwords'],
        },
    }
    
    results_file = output_dir / "tokenizer_comparison_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 分析结果已保存到: {results_file}")
    print()
    
    print("=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
