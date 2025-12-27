# Midterm Project Report: Chinese to English Neural Machine Translation

## Introduction
This report summarizes the midterm project for the course on Neural Machine Translation (NMT), focusing on the implementation of both RNN and Transformer models for translating Chinese text to English. The project aims to evaluate the performance of these models on a parallel dataset.

## Dataset
The dataset used for this project is a Chinese-English parallel corpus, which consists of sentence pairs in both languages. The data is organized as follows:
- **Training Set**: Contains a large number of sentence pairs for training the models.
- **Validation Set**: Used to tune model parameters and prevent overfitting.
- **Test Set**: Used to evaluate the final performance of the models.

The dataset files are located in the `data/raw/AP0004_Midterm&Final_translation_dataset_zh_en/` directory.

## Model Implementations
### RNN Model
The RNN model is implemented using GRU cells with an attention mechanism. The architecture consists of:
- An embedding layer to convert words into dense vectors.
- A GRU layer for sequence processing.
- An attention layer to focus on relevant parts of the input sequence.
- A fully connected layer to produce the output probabilities.

#### Training
The RNN model is trained using the `src/rnn/train.py` script. The training process involves:
- Loading the dataset.
- Preprocessing the data (tokenization, padding).
- Training the model using cross-entropy loss and an optimizer (e.g., Adam).

### Transformer Model
The Transformer model is implemented following the original architecture proposed by Vaswani et al. It includes:
- Multi-head self-attention mechanisms.
- Positional encoding to retain the order of the sequence.
- Feed-forward neural networks for processing.

#### Training
The Transformer model is trained using the `src/transformer/train.py` script. The training process is similar to that of the RNN model but utilizes different hyperparameters and optimizations suited for the Transformer architecture.

## Experimental Setup
The experiments were conducted using the following configurations:
- **RNN Configuration**: Defined in `experiments/configs/rnn_base.yaml`.
- **Transformer Configuration**: Defined in `experiments/configs/transformer_base.yaml`.

The training scripts (`scripts/train_rnn.sh` and `scripts/train_transformer.sh`) were used to initiate the training processes.

## Results
The evaluation metrics for both models were recorded in CSV files located in `experiments/results/`:
- **RNN Metrics**: `rnn_metrics.csv`
- **Transformer Metrics**: `transformer_metrics.csv`

The metrics include:
- BLEU Score: A measure of the quality of the translated text.
- Training and Validation Loss: To monitor overfitting.

### Comparison
A comparative analysis of the results shows that the Transformer model outperforms the RNN model in terms of BLEU score and training efficiency. The Transformer model benefits from its parallel processing capabilities and attention mechanisms, leading to better handling of long-range dependencies in the text.

## Conclusion
This midterm project successfully implemented and evaluated both RNN and Transformer models for Chinese to English translation. The results indicate that the Transformer model is more effective for this task. Future work may involve fine-tuning the models further and exploring additional architectures or techniques to improve translation quality.

## References
- Vaswani, A., et al. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS).
- BLEU: A Method for Automatic Evaluation of Machine Translation.