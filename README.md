# Quote Next-Word Prediction

End-to-end **next-word prediction** on famous quotes: train an **LSTM** language model from a CSV corpus in Jupyter, export **`tokenizer.pkl`**, **`max_len.pkl`**, and **`*.h5`**, then serve predictions through a **Streamlit** web UI.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Data** | `qoute_dataset.csv` — quote text and attributed authors |
| **Training** | `codefile.ipynb` — preprocessing, sequence dataset, Keras LSTM, persistence helpers |
| **Application** | `app.py` — loads artifacts and ranks the next token via softmax |

The model learns short-range patterns in quote-style English (lowercased, punctuation-stripped). Predictions are **in-domain**: inputs should resemble the training distribution for best results.

---

## Dataset

**File:** `qoute_dataset.csv`

| Column | Description |
|--------|-------------|
| `quote` | Full quotation text |
| `Author` | Attribution (used for analysis in the notebook; **not** fed into the tokenizer for next-word training in the provided pipeline) |

The corpus mixes lengths from one-liners to long passages and spans many authors (on the order of **~3k** rows and **~1k** unique authors in the version exported with this project).

---

## Methodology (training notebook)

The workflow in **`codefile.ipynb`** follows standard character/word-level language-modeling steps adapted for **word tokens**:

1. **Load** quotes from `qoute_dataset.csv` (`df['quote']`).
2. **Normalize** text: lowercase; remove punctuation via `str.translate` and `string.punctuation`.
3. **Tokenize** with Keras `Tokenizer` capped at **`num_words = 10_000`** (`vocab_size`). The fitted vocabulary is smaller than the cap when unique tokens are fewer than 10k (e.g. **~8.9k** words in the notebook output).
4. **Build supervised pairs** with a sliding window over each token sequence: for every prefix `seq[:i]`, predict the next token `seq[i]`. This yields on the order of **~85k** `(input, target)` samples for the saved configuration.
5. **Pad** all input prefixes to **`max_len`** (maximum prefix length over the training set — **745** in the notebook run) using **`padding='pre'`**.
6. **Encode targets** with one-hot vectors of size **`vocab_size`** (`to_categorical`) for **`categorical_crossentropy`**.
7. **Train** a sequential model (notebook includes both **SimpleRNN** and **LSTM** variants); the deployed app expects the **LSTM** checkpoint saved as HDF5.

Artifacts written by the notebook (aligned with the UI):

- `tokenizer.pkl` — fitted tokenizer  
- `max_len.pkl` — integer padding length  
- `lstm_model.h5` — trained model (local filenames such as `lstm_model (1).h5` also work if you keep a single `*.h5` in the app folder)

---

## Model architecture (LSTM)

Configured in the notebook:

| Layer | Configuration |
|-------|----------------|
| **Embedding** | `input_dim = 10_000`, `output_dim = 50`, `input_length = max_len` |
| **LSTM** | `units = 128` |
| **Dense** | `units = 10_000`, `activation = softmax` |

**Optimization:** Adam · **Loss:** categorical crossentropy · **Metric:** accuracy  

*(Complete training cells—including `fit()` hyperparameters such as epochs and batch size—live in `codefile.ipynb`; extend there before exporting a new `.h5` if you change the recipe.)*

---

## Streamlit application

**Entry point:** `app.py`

The UI:

- Discovers **one** HDF5 model file in the project directory (`*.h5`; first lexicographic name wins if several exist).
- Loads **`tokenizer.pkl`** and **`max_len.pkl`** once (cached).
- Accepts user text, tokenizes with the **same** tokenizer, pads with **`pre`** padding to **`max_len`**, and displays **top‑k** next-word candidates with scores.

### Running locally

```powershell
cd path\to\next_word_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Requirements:** see **`requirements.txt`** (TensorFlow/Keras, Streamlit, NumPy). Use a **virtual environment** on Windows to avoid conflicts with a broken global TensorFlow install.

---

## Repository layout

```
next_word_prediction/
├── app.py                 # Streamlit inference UI
├── requirements.txt       # Runtime dependencies for the app
├── codefile.ipynb         # Data prep + model training + pickle export
├── qoute_dataset.csv      # Quotes corpus (CSV)
├── tokenizer.pkl          # Produced by notebook (required for app)
├── max_len.pkl            # Produced by notebook (required for app)
└── *.h5                   # Trained Keras LSTM (required for app)
```

---

## Limitations

- **Out-of-vocabulary (OOV)** words are dropped by the tokenizer at inference time; if nothing maps to an ID, the app cannot propose a next word.
- Predictions reflect **quote-like** training data only; general prose or technical jargon may rank poorly.
- Author labels are not used by the next-word model in the documented notebook path—only the quote column feeds the tokenizer.

---

## Extending the project

- Retrain with more data or a larger `vocab_size`, then re-export **`tokenizer.pkl`**, **`max_len.pkl`**, and **`lstm_model.h5`** together so shapes stay consistent.
- Prefer saving models in the native **`.keras`** format for newer TensorFlow releases; the current UI loads **HDF5** via `load_model`.

---

## Acknowledgments

Quotations and attributions are aggregated in **`qoute_dataset.csv`** for educational use in this demonstration pipeline.
