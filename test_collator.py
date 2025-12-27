#!/usr/bin/env python3
"""Test data collator behavior."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('models/mt5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('models/mt5-small')

# Sample data (batch of 2)
data = {
    'zh': [
        '巴黎-随着经济危机不断加深和蔓延,整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况.',
        '一开始,很多人把这次危机比作1982年或1973年所发生的情况,这样得类比是令人宽心的,因为这两段时期意味着典型的周期性衰退.'
    ],
    'en': [
        'paris – as the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.',
        'at the start of the crisis, many people likened it to 1982 or 1973, which was reassuring, because both dates refer to classical cyclical downturns.'
    ]
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

print('Before collation:')
print('Example 0 labels length:', len(processed[0]['labels']))
print('Example 1 labels length:', len(processed[1]['labels']))

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Collate batch
batch = [processed[i] for i in range(2)]
collated = data_collator(batch)

print('\nAfter collation:')
print('Input IDs shape:', collated['input_ids'].shape)
print('Labels shape:', collated['labels'].shape)
print('Labels:', collated['labels'])
print('Contains -100:', (collated['labels'] == -100).any().item())

# Forward pass
outputs = model(**collated)
print('\nLoss:', outputs.loss.item())
print('Loss is NaN:', torch.isnan(outputs.loss).item())
