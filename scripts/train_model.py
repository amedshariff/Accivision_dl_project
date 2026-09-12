"""
Train Model Script
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_cnn_lstm import train_model

def main():
    parser = argparse.ArgumentParser(description="Train CNN + LSTM Model")
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/splits/validation.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(train_csv=args.train_csv, val_csv=args.val_csv, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
