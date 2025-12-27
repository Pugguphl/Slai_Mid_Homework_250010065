import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface for NMT project.")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True, help='Mode: train or infer')
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer'], required=True, help='Model type: rnn or transformer')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output', type=str, help='Path to save the output')
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'train':
        if args.model == 'rnn':
            os.system(f'python src/rnn/train.py --data {args.data} --output {args.output} --config {args.config}')
        elif args.model == 'transformer':
            os.system(f'python src/transformer/train.py --data {args.data} --output {args.output} --config {args.config}')
    elif args.mode == 'infer':
        if args.model == 'rnn':
            os.system(f'python src/rnn/infer.py --data {args.data} --output {args.output}')
        elif args.model == 'transformer':
            os.system(f'python src/transformer/infer.py --data {args.data} --output {args.output}')

if __name__ == "__main__":
    main()