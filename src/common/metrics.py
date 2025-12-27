def calculate_bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu(reference, candidate)

def calculate_accuracy(predictions, targets):
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

def calculate_loss(predictions, targets, criterion):
    return criterion(predictions, targets)

def log_metrics(metrics, log_file):
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv(log_file, index=False)