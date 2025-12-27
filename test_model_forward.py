#!/usr/bin/env python3
"""Test mT5 model forward pass."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('models/mt5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('models/mt5-small')

# Test input
text = 'translate Chinese to English: 巴黎-随着经济危机不断加深和蔓延,整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况.'
target = 'paris – as the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.'

# Tokenize
inputs = tokenizer(text, max_length=128, truncation=True, padding=True, return_tensors='pt')
targets = tokenizer(target, max_length=128, truncation=True, padding=True, return_tensors='pt')

print('Input IDs shape:', inputs['input_ids'].shape)
print('Target IDs shape:', targets['input_ids'].shape)

# Prepare labels (shift target right)
labels = targets['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100

print('Labels shape:', labels.shape)
print('Labels:', labels)

# Forward pass
outputs = model(**inputs, labels=labels)

print('\nLoss:', outputs.loss.item())
print('Logits shape:', outputs.logits.shape)
print('Loss is NaN:', torch.isnan(outputs.loss).item())
print('Loss is Inf:', torch.isinf(outputs.loss).item())

# Check gradients
loss = outputs.loss
loss.backward()

print('\nChecking gradients...')
has_nan_grad = False
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f'NaN gradient in {name}')
            has_nan_grad = True
        if torch.isinf(param.grad).any():
            print(f'Inf gradient in {name}')

if not has_nan_grad:
    print('No NaN gradients found')

# Check model parameters
print('\nChecking model parameters...')
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f'NaN parameter in {name}')
    if torch.isinf(param).any():
        print(f'Inf parameter in {name}')
