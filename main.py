import argparse
from src.utils import load_config
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_model(config)

if __name__ == '__main__':
    main()