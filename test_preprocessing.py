#!/usr/bin/env python3
"""Test data preprocessing."""

from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('models/mt5-small')

# Sample data
data = {
    'zh': ['巴黎-随着经济危机不断加深和蔓延,整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况.'],
    'en': ['paris – as the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.']
}

dataset = Dataset.from_dict(data)

def preprocess_function(examples, tokenizer, src_key='zh', tgt_key='en', max_length=128):
    """Preprocess examples for T5 training."""
    inputs = [f"translate Chinese to English: {text}" for text in examples[src_key]]
    targets = examples[tgt_key]

    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=False)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess
processed = dataset.map(
    lambda x: preprocess_function(x, tokenizer, 'zh', 'en', 128),
    batched=True,
    remove_columns=dataset.column_names,
)

print('Processed dataset:')
print(processed[0])
print('\nInput IDs:', processed[0]['input_ids'])
print('Labels:', processed[0]['labels'])
print('Labels length:', len(processed[0]['labels']))
print('Contains -100:', -100 in processed[0]['labels'])
