# zh-en-nmt-midterm Project

This project implements a Chinese to English Neural Machine Translation (NMT) system using both RNN and Transformer architectures. The goal is to provide a comprehensive framework for training, evaluating, and inferring translations from a dataset of parallel sentences.

## Project Structure

- **data/**: Contains all data-related files.
  - **raw/**: Directory for raw data, including links to the Chinese-English parallel sentence dataset.
  - **processed/**: Directory for processed data outputs.
  - **vocab/**: Directory for vocabulary files, including the tokenizer configuration.

- **src/**: Source code for the project.
  - **common/**: Common modules for dataset handling, evaluation metrics, tokenization, and utility functions.
  - **rnn/**: RNN model implementation, including training and inference scripts.
  - **transformer/**: Transformer model implementation, including training and inference scripts.
  - **t5/**: T5 pre-trained model fine-tuning for NMT.
  - **cli.py**: Command-line interface for handling user inputs and parameters.

- **scripts/**: Contains various scripts for preprocessing data, training models, and performing inference.
  
- **experiments/**: Directory for experiment-related files, including configuration files, logs, and results.
  
- **reports/**: Contains the midterm project report and figures.

- **tests/**: Unit tests for dataset processing and model functionalities.

## Environment Setup

To set up the environment, create a virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

## One-Click Inference

The project provides a unified inference script `inference.py` that supports all model types (RNN, Transformer, and T5).

### Usage

```bash
python inference.py --model_type <rnn|transformer|t5> --checkpoint <path_to_checkpoint> --split <valid|test> --beam_size <1|4|8>
```

### Examples

**RNN Model (Greedy Decoding):**
```bash
python inference.py --model_type rnn --checkpoint experiments/logs/best_rnn_model.pt --split test --beam_size 1
```

**RNN Model (Beam Search):**
```bash
python inference.py --model_type rnn --checkpoint experiments/logs/best_rnn_model.pt --split test --beam_size 4
```

**Transformer Model:**
```bash
python inference.py --model_type transformer --checkpoint experiments/logs/best_transformer_model.pt --split test --beam_size 4
```

**T5 Model:**
```bash
python inference.py --model_type t5 --checkpoint experiments/logs/t5_small_best --split test --beam_size 4
```

### Output

The inference script will:
- Display the BLEU score
- Show example translations (source, prediction, reference)
- Save predictions to `experiments/results/inference_outputs/<model_name>_<split>_beam<beam_size>.jsonl`

## Training

To train the RNN model, run the following command:

```bash
bash scripts/train_rnn.sh
```

To train the Transformer model, use:

```bash
bash scripts/train_transformer.sh
```

To fine-tune T5 for NMT:

```bash
bash scripts/train_t5.sh
```

## Evaluation

After training, you can evaluate the models using the metrics provided in the `experiments/results/` directory.

### RNN Results

Run all RNN experiments and generate comparison:

```bash
python scripts/analyze_rnn_results.py --split test
```

This will generate `experiments/results/rnn_comparison_master.csv` with results for:
- Different attention mechanisms (dot-product, multiplicative, additive)
- Training strategies (Teacher Forcing vs Free Running)
- Decoding strategies (Greedy vs Beam Search)

### Transformer Results

Run all Transformer ablation experiments:

```bash
python scripts/analyze_transformer_ablations.py --split test
```

This will generate `experiments/results/transformer_ablation_comparison.csv` with results for:
- Positional encoding (absolute vs relative)
- Normalization (LayerNorm vs RMSNorm)
- Hyperparameter sensitivity (batch size, learning rate, model scale)

### T5 Results

T5 fine-tuning results are saved in `experiments/results/t5_small_metrics.json`.

## Reproducibility

All scripts and configurations are designed to ensure reproducibility of experiments. Make sure to use the provided configuration files in the `experiments/configs/` directory for consistent results.

## Project Requirements

This project fulfills the following requirements:

### RNN NMT
- [x] Model structure: GRU, 2 layers, unidirectional
- [x] Attention mechanism comparison: dot-product, multiplicative, additive
- [x] Training strategy comparison: Teacher Forcing vs Free Running
- [x] Decoding strategy comparison: Greedy vs Beam Search

### Transformer NMT
- [x] Train encoder-decoder Transformer from scratch (zh→en)
- [x] Architecture ablation: positional encoding (absolute vs relative), normalization (LayerNorm vs RMSNorm)
- [x] Hyperparameter sensitivity analysis: batch size, learning rate, model scale
- [x] Pre-trained model fine-tuning: T5 for NMT

### Deliverables
- [x] Code repository + reproducible inference script
- [x] Personal project report (ID_name.pdf)
- [ ] Group presentation (10 min talk + 5 min Q&A)

**Project Report**: [250010065_庞宏林_项目报告.pdf](reports/250010065_庞宏林_项目报告.pdf)

## Acknowledgments

This project is based on the principles of Neural Machine Translation and leverages state-of-the-art architectures to achieve high-quality translations.