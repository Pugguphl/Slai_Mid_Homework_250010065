#!/usr/bin/env python3
"""比较T5和mT5的tokenizer分词结果"""

from transformers import T5Tokenizer, MT5Tokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=" * 80)
print("T5 vs mT5 Tokenizer分词比较")
print("=" * 80)
print()

print("加载tokenizers...")
t5_tokenizer = T5Tokenizer.from_pretrained("experiments/logs/t5_small_best", local_files_only=True)
mt5_tokenizer = MT5Tokenizer.from_pretrained("experiments/logs/mt5_small_best", local_files_only=True)

print(f"T5 vocab_size: {t5_tokenizer.vocab_size}")
print(f"mT5 vocab_size: {mt5_tokenizer.vocab_size}")
print()

test_texts = [
    "translate Chinese to English: 巴黎-随着经济危机不断加深和蔓延,整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况.",
    "translate Chinese to English: 中国是一个拥有五千年历史的文明古国。",
    "translate Chinese to English: 人工智能技术正在快速发展。",
    "translate Chinese to English: 我喜欢学习编程和机器学习。",
    "translate Chinese to English: 今天天气很好，适合出去散步。",
]

for i, text in enumerate(test_texts, 1):
    print(f"示例 {i}:")
    print(f"原文: {text}")
    print()
    
    t5_tokens = t5_tokenizer(text, max_length=128, truncation=True)
    mt5_tokens = mt5_tokenizer(text, max_length=128, truncation=True)
    
    print(f"T5 token数: {len(t5_tokens['input_ids'])}")
    t5_decoded = t5_tokenizer.convert_ids_to_tokens(t5_tokens['input_ids'][:30])
    print(f"T5 tokens (前30): {t5_decoded}")
    print()
    
    print(f"mT5 token数: {len(mt5_tokens['input_ids'])}")
    mt5_decoded = mt5_tokenizer.convert_ids_to_tokens(mt5_tokens['input_ids'][:30])
    print(f"mT5 tokens (前30): {mt5_decoded}")
    print()
    
    print(f"Token数差异: {len(t5_tokens['input_ids']) - len(mt5_tokens['input_ids'])} (T5 - mT5)")
    print("-" * 80)
    print()

print("=" * 80)
print("详细对比分析")
print("=" * 80)
print()

print("【词表大小对比】")
print(f"T5: {t5_tokenizer.vocab_size:,} tokens")
print(f"mT5: {mt5_tokenizer.vocab_size:,} tokens")
print(f"差异: {mt5_tokenizer.vocab_size - t5_tokenizer.vocab_size:,} tokens (mT5更大)")
print()

print("【特殊Token对比】")
print(f"T5 - pad: '{t5_tokenizer.pad_token}' (id={t5_tokenizer.pad_token_id})")
print(f"mT5 - pad: '{mt5_tokenizer.pad_token}' (id={mt5_tokenizer.pad_token_id})")
print()
print(f"T5 - eos: '{t5_tokenizer.eos_token}' (id={t5_tokenizer.eos_token_id})")
print(f"mT5 - eos: '{mt5_tokenizer.eos_token}' (id={mt5_tokenizer.eos_token_id})")
print()
print(f"T5 - unk: '{t5_tokenizer.unk_token}' (id={t5_tokenizer.unk_token_id})")
print(f"mT5 - unk: '{mt5_tokenizer.unk_token}' (id={mt5_tokenizer.unk_token_id})")
print()

print("【测试单个中文字符】")
test_chars = ["中", "国", "人", "工", "智", "能"]
for char in test_chars:
    t5_char_tokens = t5_tokenizer(char, add_special_tokens=False)
    mt5_char_tokens = mt5_tokenizer(char, add_special_tokens=False)
    print(f"字符 '{char}':")
    print(f"  T5: {len(t5_char_tokens['input_ids'])} token(s) - {t5_tokenizer.convert_ids_to_tokens(t5_char_tokens['input_ids'])}")
    print(f"  mT5: {len(mt5_char_tokens['input_ids'])} token(s) - {mt5_tokenizer.convert_ids_to_tokens(mt5_char_tokens['input_ids'])}")
print()

print("【测试英文单词】")
test_words = ["artificial", "intelligence", "machine", "learning", "translation"]
for word in test_words:
    t5_word_tokens = t5_tokenizer(word, add_special_tokens=False)
    mt5_word_tokens = mt5_tokenizer(word, add_special_tokens=False)
    print(f"单词 '{word}':")
    print(f"  T5: {len(t5_word_tokens['input_ids'])} token(s) - {t5_tokenizer.convert_ids_to_tokens(t5_word_tokens['input_ids'])}")
    print(f"  mT5: {len(mt5_word_tokens['input_ids'])} token(s) - {mt5_tokenizer.convert_ids_to_tokens(mt5_word_tokens['input_ids'])}")
print()
