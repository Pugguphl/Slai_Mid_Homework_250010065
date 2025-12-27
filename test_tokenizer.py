#!/usr/bin/env python3
"""Test mT5 tokenizer on sample data."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('models/mt5-small')

# Test input
text = 'translate Chinese to English: 巴黎-随着经济危机不断加深和蔓延,整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况.'
tokens = tokenizer(text, max_length=128, truncation=True)
print('Input tokens:', len(tokens['input_ids']))
print('First 20 tokens:', tokens['input_ids'][:20])
print('Decoded:', tokenizer.decode(tokens['input_ids'][:20]))

# Test target
target = 'paris – as the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.'
target_tokens = tokenizer(target, max_length=128, truncation=True)
print('\nTarget tokens:', len(target_tokens['input_ids']))
print('First 20 target tokens:', target_tokens['input_ids'][:20])
print('Decoded:', tokenizer.decode(target_tokens['input_ids'][:20]))

# Check pad token
print('\nPad token ID:', tokenizer.pad_token_id)
print('Pad token:', tokenizer.pad_token)
print('EOS token ID:', tokenizer.eos_token_id)
print('EOS token:', tokenizer.eos_token)
