"""
model.py
--------
Step 3 of the pipeline: the deep learning model itself.

A stacked LSTM is the classic, reliable choice for symbolic music
generation (this is the same architecture family used in well-known
projects like "Classical Piano Composer" and Google's early Magenta
melody RNNs). It treats music generation as a sequence-prediction
problem: given the last N tokens, predict the next one.

A GAN alternative is sketched at the bottom of this file for reference,
but GANs are notoriously harder to train stably on discrete sequence
data (music), so the LSTM is the recommended path to start with.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(vocab_size: int, seq_len: int, embedding_dim: int = 100) -> keras.Model:
    """Stacked LSTM next-token predictor.

    Input:  (batch, seq_len) integer token ids
    Output: (batch, vocab_size) softmax distribution over the next token
    """
    model = keras.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Optional: GAN sketch (SeqGAN-style). More complex to train than the LSTM
# above; included only as a starting point if you want to experiment.
# A GAN generator here is just an LSTM that outputs token probabilities,
# and the discriminator is a classifier (real music vs. generated) trained
# adversarially. Getting this to converge well typically needs techniques
# like policy-gradient training (REINFORCE) since token sampling is
# non-differentiable — this is genuinely more involved than the LSTM
# approach and not required to get good results.
# ---------------------------------------------------------------------------

def build_gan_discriminator(vocab_size: int, seq_len: int, embedding_dim: int = 100) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(128),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # real vs. fake
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
