"""
train.py
--------
Step 4 of the pipeline: train the LSTM on the preprocessed note sequences.

Usage:
    python train.py --data_dir data/processed --epochs 50 --batch_size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from model import build_lstm_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    with open(data_dir / "vocab.json") as f:
        meta = json.load(f)
    vocab_size = len(meta["vocab"])
    seq_len = meta["seq_len"]

    print(f"Loaded {len(X)} training examples | vocab_size={vocab_size} | seq_len={seq_len}")

    model = build_lstm_model(vocab_size=vocab_size, seq_len=seq_len)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "model_epoch{epoch:02d}_loss{loss:.4f}.keras"),
            monitor="loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4),
    ]

    model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = ckpt_dir / "final_model.keras"
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
