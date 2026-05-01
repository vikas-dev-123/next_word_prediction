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

### Raw corpus statistics

Figures below refer to the CSV shipped with this repository (computed directly from `qoute_dataset.csv`).

| Statistic | Value |
|-----------|------:|
| Rows (quotes) | **3,038** |
| Columns | `quote`, `Author` |
| Unique authors | **1,005** |
| Missing `quote` values | **0** |
| Mean `quote` length (characters) | **~155** |

The corpus mixes short one-liners and long passages across many authors.

### Derived ML dataset (after preprocessing)

These figures match the pipeline in **`codefile.ipynb`**: lowercase quotes, strip punctuation, Keras `Tokenizer(num_words=10_000)`, sliding-window `(prefix → next token)` pairs, **`pre`** padding to **`max_len`**.

| Quantity | Value |
|----------|------:|
| Maximum prefix length **`max_len`** | **745** |
| Supervised examples **`(X, y)`** | **85,271** |
| Padded input shape **`X_padded`** | **(85,271 × 745)** |
| Distinct tokens in **`word_index`** (notebook) | **~8,978** |
| Model output classes **`vocab_size`** | **10,000** |

---

## Metrics

### Objective during training

Configured in Keras for the LSTM (and SimpleRNN baseline in the notebook):

| Item | Setting |
|------|---------|
| **Loss** | Categorical crossentropy |
| **Optimization** | Adam |
| **Tracked metric** | Accuracy (`metrics=['accuracy']`) |

Per-epoch history was **not** checked into the notebook snapshot in this repo; after training, inspect `model.fit()` output or log **`History`** to CSV/TensorBoard for curves.

### Reference scores on bundled checkpoint

The following numbers were obtained by **rebuilding supervised pairs from the full corpus** with the **saved** `tokenizer.pkl`, `max_len.pkl`, and **`*.h5`**, then running **`model.evaluate`** on a **random subset of 5,000** pairs (**NumPy seed 42**, batch size 256). This measures consistency of the artifacts under the training recipe; it is **not** a held-out test split (the notebook does not define a separate validation/test partition).

| Metric | Value |
|--------|------:|
| **Loss** (subset) | **~2.26** |
| **Accuracy** (subset) | **~68.9%** |
| **Evaluate sample size** | **5,000** |

For a strict generalization estimate, split quotes or sequences **before** building sliding-window pairs, then report **`val_loss` / `val_accuracy`** from `model.fit(validation_data=...)`.

*`fit()` hyperparameters (epochs, batch size, callbacks) are defined in **`codefile.ipynb`**.*

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

### Diagrams

The figures below use **[Mermaid](https://mermaid.js.org/)** syntax; they render automatically on **GitHub** and in many Markdown previews.

#### Figure 1 — End-to-end pipeline (data → training → app)

```mermaid
flowchart TD
    subgraph Source["1. Data"]
        CSV[("qoute_dataset.csv<br/>quote + Author")]
    end

    subgraph Prep["2. Preprocessing"]
        Q["Quote column"]
        N["Normalize<br/>lowercase + strip punctuation"]
        T["Keras Tokenizer<br/>num_words = 10_000"]
        S["Integer sequences"]
        Q --> N --> T --> S
    end

    subgraph Supervised["3. Supervised dataset"]
        W["Sliding window<br/>prefix → next token"]
        P["pad_sequences<br/>max_len = 745, padding pre"]
        Y["One-hot targets<br/>to_categorical vocab_size"]
        W --> P --> Y
    end

    subgraph Train["4. Training"]
        M["LSTM model<br/>Adam + categorical CE"]
    end

    subgraph Save["5. Artifacts"]
        H5["*.h5 checkpoint"]
        PK1["tokenizer.pkl"]
        PK2["max_len.pkl"]
    end

    subgraph App["6. Inference"]
        UI["Streamlit app.py"]
        InText["User phrase"]
        TopK["Top-k next-word scores"]
        InText --> UI --> TopK
    end

    CSV --> Q
    S --> W
    P --> M
    Y --> M
    M --> H5
    M --> PK1
    M --> PK2
    H5 --> UI
    PK1 --> UI
    PK2 --> UI
```

#### Figure 2 — LSTM architecture (tensor shapes)

```mermaid
flowchart LR
    subgraph Input
        X["Token IDs<br/>batch × max_len<br/>max_len = 745"]
    end

    subgraph Layers["Keras Sequential"]
        E["Embedding<br/>input_dim 10_000<br/>output_dim 50"]
        L["LSTM<br/>units 128"]
        D["Dense<br/>units 10_000<br/>softmax"]
        E --> L --> D
    end

    subgraph Output
        Z["Probability vector<br/>batch × 10_000<br/>P next token"]
    end

    X --> E
    D --> Z
```

---

## Model architecture (LSTM)

Configured in the notebook (see **Figure 2** above for a compact block diagram):

| Layer | Configuration |
|-------|----------------|
| **Embedding** | `input_dim = 10_000`, `output_dim = 50`, `input_length = max_len` |
| **LSTM** | `units = 128` |
| **Dense** | `units = 10_000`, `activation = softmax` |

**Optimization:** Adam · **Loss:** categorical crossentropy · **Metric:** accuracy (see **Metrics** section for reference evaluation numbers).

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

## Deploying on Streamlit Community Cloud (`*.streamlit.app`)

### If logs say `Python 3.14.x` and TensorFlow / tensorflow-cpu fails to install

TensorFlow **does not ship wheels for Python 3.14**. Updating **`requirements.txt` alone will never fix that** while the host keeps Python 3.14.

Per Streamlit’s docs, **Python version is chosen when you deploy**. On many accounts you **cannot switch Python on an existing app**—you must **delete** the deployment and **deploy again**.

Do this:

1. Go to [share.streamlit.io](https://share.streamlit.io) → open your workspace.
2. **Delete** the broken app (remove deployment).
3. **Deploy from GitHub again** (same repo, branch, entrypoint `app.py`).
4. Before the final **Deploy**, open **Advanced settings**.
5. Set **Python version** to **3.11** (recommended) or **3.12**. **Do not use 3.14** with TensorFlow today.
6. Deploy, then wait for dependencies to install.

A short checklist with the same steps lives in **`STREAMLIT_CLOUD_PYTHON_FIX.txt`** in the repo root so it’s easy to spot on GitHub.

The **`runtime.txt`** file documents intent (`python-3.11`) for other hosts; **Streamlit Community Cloud does not use `runtime.txt` for Python**—the dashboard choice in step 5 is what matters.

### Dependencies on Cloud

`requirements.txt` uses **`tensorflow-cpu`** (appropriate for Linux CPU on Community Cloud), pinned to versions that install on **Python 3.11 / 3.12**. After you redeploy with one of those Python versions, installs should succeed.

**Rebooting only** an app that was created with Python 3.14 **does not change** the interpreter; you still need **delete + redeploy** with Advanced settings if your app is stuck on 3.14.

---

## Repository layout

```
next_word_prediction/
├── app.py                 # Streamlit inference UI
├── requirements.txt       # Runtime dependencies for the app
├── runtime.txt                       # Hint for some hosts (Cloud uses dashboard Python)
├── STREAMLIT_CLOUD_PYTHON_FIX.txt   # Read this if Streamlit Cloud keeps failing TensorFlow install
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
