"""Run locally after changing tokenizer.pkl: writes tokenizer_word_index.json for TF-free deploy."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> None:
    with open(BASE / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    payload = {
        "word_index": tok.word_index,
        "num_words": tok.num_words,
        "filters": getattr(tok, "filters", None),
        "lower": getattr(tok, "lower", True),
    }

    out = BASE / "tokenizer_word_index.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)

    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
