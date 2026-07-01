"""
generate.py
-----------
Step 5 of the pipeline: use the trained model to generate a brand-new
token sequence, then convert it back into a playable/saveable MIDI file.

Usage:
    python generate.py --model checkpoints/final_model.keras \
                        --data_dir data/processed \
                        --length 300 \
                        --temperature 1.0 \
                        --out output/generated.mid
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from tensorflow import keras
from music21 import stream, note, chord, instrument


def sample_with_temperature(preds: np.ndarray, temperature: float) -> int:
    """Sample the next token id from a softmax distribution, using
    temperature to control randomness/creativity.
    temperature < 1.0 -> safer, more repetitive/predictable
    temperature > 1.0 -> more surprising/experimental
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probas))


def tokens_to_midi(tokens: list[str], out_path: Path, step_duration: float = 0.5):
    """Convert a list of note/chord/rest tokens into a MIDI file."""
    output_stream = stream.Stream()
    output_stream.insert(0, instrument.Piano())

    offset = 0.0
    for tok in tokens:
        if tok == "BAR":
            continue  # separator token, not a musical event
        elif tok == "REST":
            r = note.Rest()
            r.offset = offset
            output_stream.insert(offset, r)
        elif "." in tok:
            # chord: pitch classes like "4.7.11"
            pitches = [int(p) for p in tok.split(".")]
            new_chord = chord.Chord(pitches)
            new_chord.offset = offset
            output_stream.insert(offset, new_chord)
        else:
            # single note like "C4"
            try:
                n = note.Note(tok)
                n.offset = offset
                output_stream.insert(offset, n)
            except Exception:
                pass  # skip anything malformed rather than crash the whole generation
        offset += step_duration

    output_stream.write("midi", fp=str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/final_model.keras")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--length", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default="output/generated.mid")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "vocab.json") as f:
        meta = json.load(f)
    vocab = meta["vocab"]
    token_to_int = meta["token_to_int"]
    int_to_token = {i: t for t, i in token_to_int.items()}
    seq_len = meta["seq_len"]

    with open(data_dir / "raw_tokens.pkl", "rb") as f:
        raw_tokens = pickle.load(f)
    encoded = [token_to_int[t] for t in raw_tokens]

    model = keras.models.load_model(args.model)

    # Pick a random seed sequence from the training data to start generation.
    start = random.randint(0, len(encoded) - seq_len - 1)
    pattern = encoded[start:start + seq_len]

    generated_ids = []
    for _ in range(args.length):
        input_seq = np.array(pattern).reshape(1, seq_len)
        preds = model.predict(input_seq, verbose=0)[0]
        next_id = sample_with_temperature(preds, args.temperature)
        generated_ids.append(next_id)
        pattern.append(next_id)
        pattern = pattern[1:]

    generated_tokens = [int_to_token[i] for i in generated_ids]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(generated_tokens, out_path)
    print(f"Generated {len(generated_tokens)} tokens -> saved MIDI to {out_path}")


if __name__ == "__main__":
    main()
