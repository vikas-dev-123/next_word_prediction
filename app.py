"""
Local web UI for LSTM next-word prediction using saved .h5 model, tokenizer.pkl, and max_len.pkl.
Run: streamlit run app.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent


def find_h5_model(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .h5 model found in {directory}")
    return candidates[0]


@st.cache_resource(show_spinner=True)
def load_artifacts():
    model_path = find_h5_model(BASE_DIR)
    tokenizer_path = BASE_DIR / "tokenizer.pkl"
    max_len_path = BASE_DIR / "max_len.pkl"

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing {tokenizer_path.name}")
    if not max_len_path.is_file():
        raise FileNotFoundError(f"Missing {max_len_path.name}")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)

    if not isinstance(max_len, int):
        max_len = int(max_len)

    model = load_model(model_path)
    index_word = {idx: w for w, idx in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word, model_path.name


def predict_next_words(
    text: str,
    model,
    tokenizer,
    max_len: int,
    index_word: dict[int, str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    clean = (text or "").strip()
    if not clean:
        return []

    seq = tokenizer.texts_to_sequences([clean])
    if not seq or not seq[0]:
        return []

    padded = pad_sequences(seq, maxlen=max_len, padding="pre")
    probs = model.predict(padded, verbose=0)[0]
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
    st.caption("LSTM model with your saved tokenizer and sequence length.")

    try:
        model, tokenizer, max_len, index_word, model_name = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model or artifacts: {e}")
        st.info(
            "Use a virtual environment and install dependencies:\n\n"
            "`python -m venv .venv`\n\n"
            "`.venv\\Scripts\\activate` (Windows)\n\n"
            "`pip install -r requirements.txt`"
        )
        return

    with st.expander("Loaded files", expanded=False):
        st.write(f"**Model:** `{model_name}`")
        st.write(f"**max_len:** `{max_len}`")
        st.write(f"**Tokenizer vocabulary:** `{len(tokenizer.word_index)}` words")

    text = st.text_area(
        "Type a phrase — the model predicts the **next** word",
        value="the quick brown fox",
        height=120,
        placeholder="Enter partial sentence…",
    )

    top_k = st.slider("How many suggestions", min_value=3, max_value=25, value=10)

    if st.button("Predict", type="primary"):
        preds = predict_next_words(
            text, model, tokenizer, max_len, index_word, top_k=top_k
        )
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
