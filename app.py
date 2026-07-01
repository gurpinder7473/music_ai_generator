"""
app.py
------
Streamlit front-end for the MIDI -> LSTM -> generated music pipeline.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this whole folder to a GitHub repo.
    2. On share.streamlit.io, create a new app pointing at app.py.
    3. Make sure requirements.txt (includes streamlit + music21 + tensorflow)
       is at the repo root so Streamlit Cloud installs it automatically.

The app lets you:
    1. Upload one or more MIDI files.
    2. Preprocess them into token sequences (music21).
    3. Train an LSTM on those sequences, right in the browser session.
    4. Generate a new MIDI file and download it.

Note: training deep models in a free Streamlit Cloud session is CPU-only
and resource-limited, so this UI defaults to small settings (short
sequences, few epochs) suitable for a quick demo. For serious training
runs, use train.py locally/on a GPU and then just use this app's
"Generate" tab with an uploaded pre-trained model.
"""

import io
import json
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from music21 import instrument, note, chord

from preprocess import extract_tokens_from_file, build_sequences
from model import build_lstm_model
from generate import sample_with_temperature, tokens_to_midi

st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="centered")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key, default in [
    ("tokens", None),
    ("vocab", None),
    ("token_to_int", None),
    ("int_to_token", None),
    ("X", None),
    ("y", None),
    ("model", None),
    ("seq_len", 60),
    ("generated_midi_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🎵 AI Music Generator")
st.caption("MIDI collection → note sequences → LSTM training → new generated MIDI, all in one app.")

tab1, tab2, tab3 = st.tabs(["1. Upload & Preprocess", "2. Train", "3. Generate"])

# ---------------------------------------------------------------------------
# Tab 1: Upload + preprocess
# ---------------------------------------------------------------------------
with tab1:
    st.header("Upload MIDI files")
    st.write(
        "Upload a handful of `.mid` / `.midi` files in a consistent style "
        "(e.g. all piano, all one composer or genre) for the most coherent results."
    )

    uploaded_files = st.file_uploader(
        "MIDI files", type=["mid", "midi"], accept_multiple_files=True
    )

    seq_len = st.slider(
        "Sequence length (how many previous notes the model sees)",
        min_value=20, max_value=150, value=60, step=10,
    )

    if st.button("Preprocess", type="primary", disabled=not uploaded_files):
        with st.spinner("Parsing MIDI files with music21..."):
            all_tokens = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for uf in uploaded_files:
                    tmp_path = Path(tmpdir) / uf.name
                    tmp_path.write_bytes(uf.getvalue())
                    file_tokens = extract_tokens_from_file(tmp_path)
                    all_tokens.extend(file_tokens)
                    all_tokens.append("BAR")

            if len(all_tokens) <= seq_len:
                st.error(
                    "Not enough note data extracted to build a training sequence. "
                    "Try uploading more/longer MIDI files or reducing sequence length."
                )
            else:
                X, y, vocab, token_to_int = build_sequences(all_tokens, seq_len)
                st.session_state.tokens = all_tokens
                st.session_state.vocab = vocab
                st.session_state.token_to_int = token_to_int
                st.session_state.int_to_token = {i: t for t, i in token_to_int.items()}
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.seq_len = seq_len
                st.session_state.model = None  # invalidate any old trained model

                st.success(
                    f"Extracted {len(all_tokens)} tokens, vocabulary of "
                    f"{len(vocab)} unique notes/chords, {len(X)} training examples."
                )

    if st.session_state.X is not None:
        st.info(
            f"Ready to train: {len(st.session_state.X)} sequences, "
            f"vocab size {len(st.session_state.vocab)}, seq_len {st.session_state.seq_len}."
        )

# ---------------------------------------------------------------------------
# Tab 2: Train
# ---------------------------------------------------------------------------
with tab2:
    st.header("Train the LSTM")

    if st.session_state.X is None:
        st.warning("Preprocess some MIDI files in tab 1 first.")
    else:
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=20)
        batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=64)

        st.caption(
            "Free CPU sessions are slow — start small (10-20 epochs) to confirm "
            "everything works, then increase if you have time/GPU."
        )

        if st.button("Train model", type="primary"):
            vocab_size = len(st.session_state.vocab)
            model = build_lstm_model(vocab_size=vocab_size, seq_len=st.session_state.seq_len)

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_chart = st.line_chart()

            class StreamlitCallback:
                pass

            from tensorflow import keras

            class StCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    frac = (epoch + 1) / epochs
                    progress_bar.progress(min(frac, 1.0))
                    status_text.text(
                        f"Epoch {epoch + 1}/{epochs} — loss: {logs['loss']:.4f}, "
                        f"accuracy: {logs.get('accuracy', 0):.4f}"
                    )
                    loss_chart.add_rows({"loss": [logs["loss"]]})

            with st.spinner("Training..."):
                model.fit(
                    st.session_state.X, st.session_state.y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[StCallback()],
                )

            st.session_state.model = model
            st.success("Training complete! Head to tab 3 to generate music.")

    if st.session_state.model is not None:
        st.info("A trained model is loaded and ready for generation.")

# ---------------------------------------------------------------------------
# Tab 3: Generate
# ---------------------------------------------------------------------------
with tab3:
    st.header("Generate new music")

    if st.session_state.model is None:
        st.warning("Train a model in tab 2 first.")
    else:
        gen_length = st.slider("Number of notes to generate", 50, 1000, 300, step=50)
        temperature = st.slider(
            "Temperature (creativity)", 0.2, 2.0, 1.0, step=0.1,
            help="Lower = safer/more repetitive. Higher = more surprising/experimental.",
        )

        if st.button("Generate", type="primary"):
            with st.spinner("Generating..."):
                encoded = [st.session_state.token_to_int[t] for t in st.session_state.tokens]
                seq_len = st.session_state.seq_len
                start = random.randint(0, len(encoded) - seq_len - 1)
                pattern = encoded[start:start + seq_len]

                generated_ids = []
                for _ in range(gen_length):
                    input_seq = np.array(pattern).reshape(1, seq_len)
                    preds = st.session_state.model.predict(input_seq, verbose=0)[0]
                    next_id = sample_with_temperature(preds, temperature)
                    generated_ids.append(next_id)
                    pattern.append(next_id)
                    pattern = pattern[1:]

                generated_tokens = [st.session_state.int_to_token[i] for i in generated_ids]

                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / "generated.mid"
                    tokens_to_midi(generated_tokens, out_path)
                    midi_bytes = out_path.read_bytes()

                st.session_state.generated_midi_bytes = midi_bytes
                st.success(f"Generated {len(generated_tokens)} notes/chords.")

        if st.session_state.generated_midi_bytes is not None:
            st.download_button(
                label="⬇️ Download generated.mid",
                data=st.session_state.generated_midi_bytes,
                file_name="generated.mid",
                mime="audio/midi",
            )
            st.caption(
                "Browsers can't play raw MIDI directly. Open the downloaded file in "
                "a DAW (GarageBand, MuseScore, Ableton) or convert it to WAV/MP3 "
                "with a SoundFont synth (e.g. fluidsynth) to listen."
            )
