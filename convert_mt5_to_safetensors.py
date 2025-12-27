import torch
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import T5Config

print('Loading model checkpoint...')
checkpoint = torch.load('models/mt5-small/pytorch_model.bin', map_location='cpu', weights_only=False)
print(f'Checkpoint loaded with {len(checkpoint)} keys')

print('Loading model configuration...')
config = T5Config.from_pretrained('models/mt5-small')
print(f'Config loaded: {config.model_type}')

print('Creating model instance...')
model = T5ForConditionalGeneration(config)
print('Model instance created')

print('Loading state dict...')
model.load_state_dict(checkpoint)
print('State dict loaded')

print('Saving model in safetensors format...')
model.save_pretrained('models/mt5-small', safe_serialization=True)
print('Model saved successfully in safetensors format!')
