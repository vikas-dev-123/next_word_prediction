"""
Next-word prediction UI: uses TensorFlow when installed; otherwise NumPy + h5py
(so Streamlit Cloud on Python 3.14 can run without TensorFlow wheels).

Run: streamlit run app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


class TfBundle(TypedDict):
    backend: str
    model: Any
    tokenizer: Any
    max_len: int
    index_word: dict[int, str]
    model_name: str


class NpBundle(TypedDict):
    backend: str
    weights: dict[str, np.ndarray]
    word_index: dict[str, int]
    num_words: int
    filters: str
    lower: bool
    max_len: int
    lstm_units: int
    index_word: dict[int, str]
    model_name: str


ArtifactBundle = TfBundle | NpBundle


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def find_h5_model(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .h5 model found in {directory}")
    return candidates[0]


@st.cache_resource(show_spinner=True)
def load_artifacts() -> ArtifactBundle:
    model_path = find_h5_model(BASE_DIR)
    tokenizer_pkl = BASE_DIR / "tokenizer.pkl"
    tokenizer_json = BASE_DIR / "tokenizer_word_index.json"
    max_len_path = BASE_DIR / "max_len.pkl"

    if not max_len_path.is_file():
        raise FileNotFoundError(f"Missing {max_len_path.name}")

    with open(max_len_path, "rb") as f:
        raw_ml = pickle.load(f)
    max_len = int(raw_ml)

    if _tensorflow_available() and tokenizer_pkl.is_file():
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        from tensorflow.keras.models import load_model

        with open(tokenizer_pkl, "rb") as f:
            tokenizer = pickle.load(f)
        model = load_model(model_path)
        index_word = {idx: w for w, idx in tokenizer.word_index.items()}
        return TfBundle(
            backend="tf",
            model=model,
            tokenizer=tokenizer,
            max_len=max_len,
            index_word=index_word,
            model_name=model_path.name,
        )

    if not tokenizer_json.is_file():
        raise FileNotFoundError(
            f"TensorFlow is not installed and {tokenizer_json.name} is missing. "
            "Run `python export_tokenizer_json.py` locally and commit the JSON file."
        )

    from numpy_inference import (
        load_keras_h5_weights,
        load_tokenizer_json,
        lstm_units_from_weights,
    )

    wi, nw, filters, lower = load_tokenizer_json(tokenizer_json)
    weights = load_keras_h5_weights(model_path)
    units = lstm_units_from_weights(weights)
    index_word = {idx: w for w, idx in wi.items()}

    return NpBundle(
        backend="np",
        weights=weights,
        word_index=wi,
        num_words=nw,
        filters=filters,
        lower=lower,
        max_len=max_len,
        lstm_units=units,
        index_word=index_word,
        model_name=model_path.name,
    )


def predict_next_words(
    text: str,
    bundle: ArtifactBundle,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    clean = (text or "").strip()
    if not clean:
        return []

    if bundle["backend"] == "tf":
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        max_len = bundle["max_len"]
        index_word = bundle["index_word"]

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        seq = tokenizer.texts_to_sequences([clean])
        if not seq or not seq[0]:
            return []
        padded = pad_sequences(seq, maxlen=max_len, padding="pre")
        probs = model.predict(padded, verbose=0)[0]
    else:
        from numpy_inference import pad_pre, predict_probs_numpy, texts_to_sequences

        seq = texts_to_sequences(
            bundle["word_index"],
            bundle["num_words"],
            clean,
            filters=bundle["filters"],
            lower=bundle["lower"],
        )
        if not seq:
            return []
        padded_arr = pad_pre(seq, bundle["max_len"])
        probs = predict_probs_numpy(
            padded_arr,
            bundle["weights"],
            bundle["lstm_units"],
        )
        index_word = bundle["index_word"]

    top_indices = np.argsort(probs)[-top_k:][::-1]

    out: list[tuple[str, float]] = []
    for idx in top_indices:
        idx = int(idx)
        word = index_word.get(idx)
        if word is None:
            continue
        out.append((word, float(probs[idx])))
        if len(out) >= top_k:
            break
    return out


def main():
    st.set_page_config(
        page_title="Next-word prediction",
        page_icon="✎",
        layout="centered",
    )

    st.title("Next-word prediction")
    st.caption(
        "LSTM model with your tokenizer and sequence length "
        "(TensorFlow if available; otherwise NumPy + h5py)."
    )

    try:
        bundle = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model or artifacts: {e}")
        st.info(
            "Install deps: `pip install -r requirements.txt` "
            "(optional TensorFlow: `pip install -r requirements-local.txt`). "
            "Cloud deploy needs `tokenizer_word_index.json` — run "
            "`python export_tokenizer_json.py` locally and push it."
        )
        return

    backend_label = "TensorFlow" if bundle["backend"] == "tf" else "NumPy (no TensorFlow)"
    with st.expander("Loaded files", expanded=False):
        st.write(f"**Model:** `{bundle['model_name']}`")
        st.write(f"**Backend:** `{backend_label}`")
        st.write(f"**max_len:** `{bundle['max_len']}`")
        st.write(f"**Tokenizer vocabulary:** `{len(bundle['index_word'])}` words")

    text = st.text_area(
        "Type a phrase — the model predicts the **next** word",
        value="the quick brown fox",
        height=120,
        placeholder="Enter partial sentence…",
    )

    top_k = st.slider("How many suggestions", min_value=3, max_value=25, value=10)

    if st.button("Predict", type="primary"):
        preds = predict_next_words(text, bundle, top_k=top_k)
        if not preds:
            st.warning(
                "No prediction: empty input or no known tokens in your phrase "
                "(words must appear in the tokenizer vocabulary)."
            )
        else:
            st.subheader("Top predictions")
            for rank, (word, p) in enumerate(preds, start=1):
                st.markdown(f"**{rank}.** `{word}` — {p:.4f}")
                st.progress(min(max(p, 0.0), 1.0))


if __name__ == "__main__":
    main()
