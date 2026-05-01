"""
Pure NumPy + h5py inference for the bundled Keras H5 LSTM (no TensorFlow).
Used when TensorFlow cannot be installed (e.g. Streamlit Cloud on Python 3.14).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_final_hidden(
    x_seq: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray,
    units: int,
) -> np.ndarray:
    """Process full sequence; return last hidden state (matches return_sequences=False)."""
    ki = kernel[:, :units]
    kf = kernel[:, units : 2 * units]
    kc = kernel[:, 2 * units : 3 * units]
    ko = kernel[:, 3 * units :]

    ri = recurrent_kernel[:, :units]
    rf = recurrent_kernel[:, units : 2 * units]
    rc = recurrent_kernel[:, 2 * units : 3 * units]
    ro = recurrent_kernel[:, 3 * units :]

    bi = bias[:units]
    bf = bias[units : 2 * units]
    bc = bias[2 * units : 3 * units]
    bo = bias[3 * units :]

    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)

    for t in range(x_seq.shape[0]):
        x = x_seq[t].astype(np.float32)
        z_i = x @ ki + h @ ri + bi
        z_f = x @ kf + h @ rf + bf
        z_c = x @ kc + h @ rc + bc
        z_o = x @ ko + h @ ro + bo

        i = _sigmoid(z_i)
        f = _sigmoid(z_f)
        c_tilde = np.tanh(z_c)
        o = _sigmoid(z_o)

        c = f * c + i * c_tilde
        h = o * np.tanh(c)

    return h.astype(np.float32)


def load_keras_h5_weights(h5_path: Path) -> dict[str, np.ndarray]:
    """Load weight arrays from this project's Keras Sequential HDF5 export."""
    paths = {
        "embeddings": "model_weights/embedding_1/sequential_1/embedding_1/embeddings",
        "lstm_kernel": "model_weights/lstm/sequential_1/lstm/lstm_cell/kernel",
        "lstm_recurrent": "model_weights/lstm/sequential_1/lstm/lstm_cell/recurrent_kernel",
        "lstm_bias": "model_weights/lstm/sequential_1/lstm/lstm_cell/bias",
        "dense_kernel": "model_weights/dense_1/sequential_1/dense_1/kernel",
        "dense_bias": "model_weights/dense_1/sequential_1/dense_1/bias",
    }
    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for k, p in paths.items():
            out[k] = np.asarray(f[p][:], dtype=np.float32)
    return out


_KERAS_DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


def load_tokenizer_json(
    json_path: Path,
) -> tuple[dict[str, int], int, str, bool]:
    with open(json_path, encoding="utf-8") as fp:
        obj = json.load(fp)
    word_index = {str(k): int(v) for k, v in obj["word_index"].items()}
    num_words = int(obj["num_words"])
    filters = str(obj.get("filters") or _KERAS_DEFAULT_FILTERS)
    lower = bool(obj.get("lower", True))
    return word_index, num_words, filters, lower


def texts_to_sequences(
    word_index: dict[str, int],
    num_words: int,
    text: str,
    *,
    filters: str = _KERAS_DEFAULT_FILTERS,
    lower: bool = True,
) -> list[int]:
    """Match `keras.preprocessing.text.Tokenizer.texts_to_sequences` preprocessing."""

    raw = (text or "").strip()
    if lower:
        raw = raw.lower()

    translator = str.maketrans(filters, " " * len(filters))
    clean = raw.translate(translator)
    seq: list[int] = []
    for w in clean.split():
        idx = word_index.get(w)
        if idx is None:
            continue
        if num_words and idx >= num_words:
            continue
        seq.append(idx)
    return seq


def pad_pre(seq: list[int], maxlen: int) -> np.ndarray:
    arr = np.zeros(maxlen, dtype=np.int32)
    if not seq:
        return arr
    take = seq[-maxlen:]
    if len(take) < maxlen:
        arr[-len(take) :] = take
    else:
        arr[:] = take
    return arr


def predict_probs_numpy(
    padded_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    lstm_units: int,
) -> np.ndarray:
    emb = weights["embeddings"][padded_ids]
    h = _lstm_final_hidden(
        emb,
        weights["lstm_kernel"],
        weights["lstm_recurrent"],
        weights["lstm_bias"],
        lstm_units,
    )
    logits = h @ weights["dense_kernel"] + weights["dense_bias"]
    logits -= np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def lstm_units_from_weights(weights: dict[str, np.ndarray]) -> int:
    lk = weights["lstm_kernel"]
    return lk.shape[1] // 4
