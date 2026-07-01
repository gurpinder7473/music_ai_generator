"""
preprocess.py
-------------
Step 2 of the pipeline: turn a folder of MIDI files into note/chord
sequences that a neural network can learn from.

Usage:
    python preprocess.py --midi_dir data/midi --out_dir data/processed

What it does:
    1. Walks every .mid / .midi file in --midi_dir.
    2. Uses music21 to parse each file into a flat stream of notes and chords.
       - A single note becomes a string like "C4" (pitch name + octave).
       - A chord becomes a string like "4.7.11" (pitch classes joined by '.').
       - Rests become the token "REST".
    3. Builds a vocabulary mapping every unique token -> integer id.
    4. Slices the resulting token stream into overlapping fixed-length
       sequences (X) with the next token as the target (y), the standard
       "predict the next note" framing used for music LSTMs.
    5. Saves everything (sequences, vocabulary, metadata) to --out_dir so
       train.py can load it without re-parsing MIDI every time.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from music21 import converter, instrument, note, chord
except ImportError:
    raise SystemExit(
        "music21 is not installed. Run: pip install -r requirements.txt "
        "(use --break-system-packages if needed)"
    )

SEQUENCE_LENGTH = 100  # how many previous tokens the model sees before predicting the next one


def extract_tokens_from_file(midi_path: Path) -> list[str]:
    """Parse a single MIDI file into a list of note/chord/rest tokens."""
    tokens = []
    try:
        midi_stream = converter.parse(midi_path)
    except Exception as e:
        print(f"  [skip] could not parse {midi_path.name}: {e}")
        return tokens

    # Try to isolate a single instrument part (piano etc.) if the file
    # has multiple parts; otherwise fall back to the flat stream.
    try:
        parts = instrument.partitionByInstrument(midi_stream)
    except Exception:
        parts = None

    elements = parts.parts[0].recurse() if parts and len(parts.parts) > 0 else midi_stream.flat

    for el in elements:
        if isinstance(el, note.Note):
            tokens.append(str(el.pitch))
        elif isinstance(el, chord.Chord):
            tokens.append(".".join(str(p) for p in el.normalOrder))
        elif isinstance(el, note.Rest):
            tokens.append("REST")

    return tokens


def build_sequences(all_tokens: list[str], seq_len: int):
    """Build the vocabulary and (X, y) integer sequences."""
    vocab = sorted(set(all_tokens))
    token_to_int = {tok: i for i, tok in enumerate(vocab)}

    encoded = [token_to_int[t] for t in all_tokens]

    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    return X, y, vocab, token_to_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", default="data/midi", help="Folder containing .mid/.midi files")
    parser.add_argument("--out_dir", default="data/processed", help="Where to save processed arrays")
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi")))
    if not midi_files:
        raise SystemExit(
            f"No .mid/.midi files found in {midi_dir}. "
            "Add some MIDI files there first (see README for sources)."
        )

    print(f"Found {len(midi_files)} MIDI files. Extracting notes/chords...")
    all_tokens: list[str] = []
    for f in tqdm(midi_files):
        all_tokens.extend(extract_tokens_from_file(f))
        all_tokens.append("BAR")  # simple separator token between songs

    if len(all_tokens) <= args.seq_len:
        raise SystemExit("Not enough note data extracted to build even one training sequence.")

    print(f"Extracted {len(all_tokens)} tokens total. Building training sequences (len={args.seq_len})...")
    X, y, vocab, token_to_int = build_sequences(all_tokens, args.seq_len)
    print(f"Vocabulary size: {len(vocab)} unique tokens. Training examples: {len(X)}")

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump({"vocab": vocab, "token_to_int": token_to_int, "seq_len": args.seq_len}, f)
    with open(out_dir / "raw_tokens.pkl", "wb") as f:
        pickle.dump(all_tokens, f)

    print(f"Done. Saved processed data to {out_dir}/")


if __name__ == "__main__":
    main()
