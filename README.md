# FairGrade AI

**Bias-aware tabular ML demo** — compare a high-performance **XGBoost**...

Built for learning and portfolio use: transparent feature engineering, held-out fairness metrics, and a simple **Streamlit** UI to explore individual profiles.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://fairgrade-ai-qkjkihj6sgjufuwqoaxddh.streamlit.app)
---

## What this project does

1. **Prepare data** (`data_prep.py`) — loads the public [UCI Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance) (Mathematics) CSV, engineers numeric features, one-hot encodes job/reason fields, and defines a binary **pass/fail** target from final grade `G3 ≥ 10`.
2. **Train two models** (`train.py`) — XGBoost vs. a dropout + AdamW-regularized neural network on identical inputs.
3. **Compare in the browser** (`app.py`) — enter a student profile and see both predictions plus hold-out **accuracy** and **disparate impact**.

**Important:** `address` (Urban `U` / Rural `R`) is **never** fed to either model. It is used only offline to measure whether predicted pass rates differ by geography on the test split.

---

## Tech stack

| Layer | Tools |
|--------|--------|
| Data | [pandas](https://pandas.pydata.org/), [pathlib](https://docs.python.org/3/library/pathlib.html) |
| Baseline model | [XGBoost](https://xgboost.readthedocs.io/) (`XGBClassifier`) |
| Fair model | [PyTorch](https://pytorch.org/) MLP with **Dropout** + **AdamW** weight decay |
| Preprocessing | [scikit-learn](https://scikit-learn.org/) `StandardScaler`, stratified split |
| UI | [Streamlit](https://streamlit.io/) |
| Metrics | Custom disparate impact + sklearn `accuracy_score` |

---

## Fairness metric: Disparate Impact (DI)

On the hold-out set, we compare predicted **pass** rates (`ŷ = 1`) between Urban and Rural students (from the `address` column):

$$
\text{DI} = \frac{P(\hat{y} = 1 \mid \text{Urban})}{P(\hat{y} = 1 \mid \text{Rural})}
$$

| DI value | Interpretation (on this metric) |
|----------|----------------------------------|
| **1.0** | Equal predicted pass rates (parity) |
| **> 1.0** | Urban students predicted to pass more often |
| **< 1.0** | Rural students predicted to pass more often |

DI is a common **group fairness diagnostic**; it does not replace legal, ethical, or domain-specific review. Models here are for **research and education**, not production grading decisions.

---

## Repository layout

```
fairgrade-ai/
├── app.py                 # Streamlit dashboard
├── data_prep.py           # Feature engineering and loading
├── train.py               # Training, evaluation, artifact export
├── data/
│   └── student-mat.csv    # UCI dataset (semicolon-separated)
├── requirements.txt
├── README.md
└── .gitignore
```

After training (not committed by default):

- `baseline_model.pkl` — XGBoost + feature column list  
- `fair_model.pt` — PyTorch weights, scaler, schema  
- `metrics.json` — hold-out accuracy and DI for both models  

---

## Setup

### Prerequisites

- Python **3.10+** recommended  
- Git  

### 1. Clone and enter the project

```bash
git clone https://github.com/<your-username>/fairgrade-ai.git
cd fairgrade-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify data

Ensure `data/student-mat.csv` exists (included in this repo). It is the UCI Student Performance file with `;` separators.

---

## Usage

Run all commands from the **project root** (where `train.py` lives).

### Prepare / inspect data (optional)

```bash
python data_prep.py
```

### Train models and write metrics

```bash
python train.py
```

This prints a **Bias Crash Test** table and creates `baseline_model.pkl`, `fair_model.pt`, and `metrics.json`.

### Launch the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`), set sidebar fields, and click **Predict**.

---

## Design notes

- **`drop_first=True` in one-hot encoding** — avoids redundant dummy columns (dummy variable trap) and keeps the feature matrix identifiable.
- **Dropout + weight decay** — reduces overfitting on small tabular data; encourages smoother decision boundaries that often yield DI closer to 1.0 than aggressive boosting alone.
- **Stratified train/test split** — preserves pass/fail ratio in both splits (`random_state=42`).
- **Inference alignment** — `app.py` uses the same `build_feature_matrix` + `align_to_training_columns` path as training so column order and missing dummies stay consistent.

---

## Security and privacy

- Uses the **public UCI dataset** only (no real student PII in this pipeline).
- **No API keys, passwords, or machine-specific paths** in source code.
- **Do not commit** `.streamlit/secrets.toml` or untrusted pickle/checkpoint files.
- Load `baseline_model.pkl` / `fair_model.pt` only from artifacts you trained locally.

---

## License

Dataset terms follow the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) usage policy.

---

