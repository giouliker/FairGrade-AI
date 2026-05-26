"""
FairGrade AI — training pipeline (XGBoost baseline vs. regularized PyTorch MLP).

Trains two classifiers on the same engineered features, evaluates accuracy and
disparate impact on a stratified hold-out set, and writes artifacts consumed
by ``app.py``.

Security note: artifacts are written with ``pickle`` / ``torch.save`` to paths
relative to the repository root. Only load these files from trusted local runs
of this script (never unpickle checkpoints from untrusted sources).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from data_prep import DATA_PATH, load_and_preprocess_data
from paths import (
    BASELINE_MODEL_PATH,
    FAIR_MODEL_PATH,
    METRICS_PATH,
    ensure_models_dir,
)

# Re-exported for app.py and documentation.
__all__ = [
    "BASELINE_MODEL_PATH",
    "FAIR_MODEL_PATH",
    "METRICS_PATH",
    "PassPredictor",
    "predict_neural_network",
    "main",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# UCI ``address`` field codes used only for fairness evaluation.
RURAL_LABEL = "R"
URBAN_LABEL = "U"

# XGBoost: strong default tabular booster; shallow trees limit extreme splits.
XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 100,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    # Single-threaded fit avoids fork/spawn issues on headless hosts (e.g. Streamlit Cloud).
    "n_jobs": 1,
}

# Neural network: small MLP suited to ~400 samples and ~20+ features.
NN_HIDDEN_SIZE = 64
NN_DROPOUT = 0.2
# Override with NN_EPOCHS env var on slow cloud instances (default tuned for CI/cloud).
NN_EPOCHS = int(os.environ.get("NN_EPOCHS", "80"))
NN_BATCH_SIZE = 32
NN_LEARNING_RATE = 1e-3
# AdamW weight decay penalizes large weights (L2-style), complementing Dropout.
NN_WEIGHT_DECAY = 1e-4


def compute_disparate_impact(
    y_pred: np.ndarray,
    protected: pd.Series,
    privileged_label: str = URBAN_LABEL,
    unprivileged_label: str = RURAL_LABEL,
) -> float:
    """
    Compute disparate impact (DI) between Urban and Rural predicted pass rates.

    Formula
    -------
    .. code-block:: text

        DI = P(pass | Urban) / P(pass | Rural)

    where ``pass`` means the model predicted label ``1``.

    Interpretation
    --------------
    - **DI = 1.0**: equal predicted pass rates (parity on this metric).
    - **DI > 1.0**: urban students receive favorable predictions more often.
    - **DI < 1.0**: rural students receive favorable predictions more often.

    This is a common *group fairness* diagnostic. It does not replace legal
    or domain-specific fairness requirements, but it quantifies whether the
    model's positive rate differs by ``address`` on the evaluation split.

    Parameters
    ----------
    y_pred:
        Binary predictions (0/1) on the test set.
    protected:
        Series of ``'U'`` / ``'R'`` address labels aligned with ``y_pred``.
    privileged_label:
        Numerator group (default Urban ``'U'``).
    unprivileged_label:
        Denominator group (default Rural ``'R'``).

    Returns
    -------
    float
        Disparate impact ratio, or ``nan`` / ``inf`` when denominators vanish.
    """
    protected = np.asarray(protected)
    y_pred = np.asarray(y_pred, dtype=int)

    rural_mask = protected == unprivileged_label
    urban_mask = protected == privileged_label

    rural_rate = float(y_pred[rural_mask].mean()) if rural_mask.any() else np.nan
    urban_rate = float(y_pred[urban_mask].mean()) if urban_mask.any() else np.nan

    if rural_rate == 0:
        return np.nan if urban_rate == 0 else np.inf
    return urban_rate / rural_rate


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    address_test: pd.Series,
) -> dict[str, float]:
    """
    Summarize one model on the hold-out split.

    Returns accuracy (sklearn) and disparate impact (Urban/Rural DI).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "disparate_impact": float(compute_disparate_impact(y_pred, address_test)),
    }


def print_bias_crash_test_table(
    xgb_metrics: dict[str, float],
    nn_metrics: dict[str, float],
) -> None:
    """Print a side-by-side CLI table comparing baseline vs. fair model metrics."""
    print("\n" + "=" * 62)
    print("  BIAS CRASH TEST — Test Set Metrics (Urban / Rural DI)")
    print("=" * 62)
    print(f"{'Metric':<22} {'XGBoost (Baseline)':>18} {'Neural Net (Fair)':>18}")
    print("-" * 62)
    print(
        f"{'Accuracy':<22} "
        f"{xgb_metrics['accuracy']:>18.4f} "
        f"{nn_metrics['accuracy']:>18.4f}"
    )
    print(
        f"{'Disparate Impact':<22} "
        f"{xgb_metrics['disparate_impact']:>18.4f} "
        f"{nn_metrics['disparate_impact']:>18.4f}"
    )
    print("-" * 62)
    print("DI = urban pass rate / rural pass rate  (1.0 = parity)")

    xgb_gap = abs(xgb_metrics["disparate_impact"] - 1.0)
    nn_gap = abs(nn_metrics["disparate_impact"] - 1.0)
    fairer = "Neural Network" if nn_gap < xgb_gap else "XGBoost"
    print(f"\nFairer model (DI closer to 1.0): {fairer}")
    print(
        "Note: gradient boosting can fit sharp splits correlated with proxies "
        "for geography; Dropout + AdamW encourage smoother neural boundaries.\n"
    )


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    address: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    80/20 stratified split on ``y`` so pass/fail prevalence is stable in both sets.

    ``address`` is split in parallel for fairness metrics only.
    """
    return train_test_split(
        X,
        y,
        address,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """
    Train the XGBoost tabular baseline.

    XGBoost is included as a high-accuracy reference model for structured data.
    Sequential boosting can emphasize rare high-gain splits; on biased historical
    labels, that may widen Urban/Rural prediction gaps (measured via DI).
    """
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model


def save_baseline_artifact(
    model: XGBClassifier,
    feature_columns: list[str],
    path: Path = BASELINE_MODEL_PATH,
) -> None:
    """Save XGBoost model plus feature column order for ``app.py`` inference."""
    ensure_models_dir()
    artifact = {"model": model, "feature_columns": feature_columns}
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved XGBoost baseline to {path}")


class PassPredictor(nn.Module):
    """
    Three-layer MLP for binary pass/fail prediction (logits output).

    Regularization choices
    ----------------------
    **Dropout** (after each hidden ReLU):
        Randomly zeroes activations during training so the network cannot rely
        on any single neuron. On small tabular data this reduces memorization
        of spurious patterns (including those correlated with geography in
        the training labels).

    **AdamW weight decay** (configured in :func:`train_neural_network`):
        Adds an L2 penalty on weights during optimization, encouraging smaller
        coefficients and smoother decision boundaries than an unregularized MLP.

    Together, Dropout and weight decay target **generalization** and often
    improve parity metrics (DI closer to 1.0) relative to an aggressive booster.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = NN_HIDDEN_SIZE,
        dropout: float = NN_DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for BCEWithLogitsLoss (no sigmoid inside the module)."""
        return self.net(x).squeeze(-1)


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[PassPredictor, StandardScaler]:
    """
    Train the fair MLP with feature scaling and regularization.

    Steps
    -----
    1. ``StandardScaler`` on numeric+one-hot features (zero mean, unit variance).
    2. Mini-batch training with ``BCEWithLogitsLoss``.
    3. ``AdamW`` optimizer with ``weight_decay=NN_WEIGHT_DECAY``.
    4. ``Dropout`` active inside :class:`PassPredictor` during ``model.train()``.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=NN_BATCH_SIZE,
        shuffle=True,
    )

    model = PassPredictor(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_LEARNING_RATE,
        weight_decay=NN_WEIGHT_DECAY,
    )

    model.train()
    for epoch in range(NN_EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{NN_EPOCHS} — loss: {epoch_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
        test_acc = (test_preds == y_test_t).float().mean().item()
    print(f"Neural network training complete (test accuracy: {test_acc:.4f})")

    return model, scaler


def predict_neural_network(
    model: PassPredictor,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return binary pass/fail predictions (threshold 0.5 on sigmoid logits)."""
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        return (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)


def save_fair_artifact(
    model: PassPredictor,
    scaler: StandardScaler,
    feature_columns: list[str],
    path: Path = FAIR_MODEL_PATH,
) -> None:
    """
    Persist neural network weights, fitted scaler, and feature schema.

    ``app.py`` reloads ``hidden_size`` and ``dropout`` so architecture matches
    the checkpoint even if hyperparameters change in a future training run.
    """
    ensure_models_dir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "feature_columns": feature_columns,
            "input_dim": len(feature_columns),
            "hidden_size": NN_HIDDEN_SIZE,
            "dropout": NN_DROPOUT,
        },
        path,
    )
    print(f"Saved fair neural network to {path}")


def save_metrics(
    xgb_metrics: dict[str, float],
    nn_metrics: dict[str, float],
    path: Path = METRICS_PATH,
) -> None:
    """Write hold-out metrics JSON for the Streamlit dashboard."""
    ensure_models_dir()
    payload = {"baseline": xgb_metrics, "fair": nn_metrics}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {path}")


def main() -> None:
    """Run the full training and evaluation pipeline."""
    ensure_models_dir()
    if not DATA_PATH.is_file():
        print(f"ERROR: Dataset missing at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    # Headless-safe defaults (no GUI backends required).
    X, y, address = load_and_preprocess_data()
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test, _, address_test = stratified_train_test_split(
        X, y, address
    )
    address_test = address_test.reset_index(drop=True)

    print(f"Stratified split — train: {len(X_train)}, test: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}\n")

    print("Training XGBoost baseline...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_preds, address_test)

    print("\nTraining regularized neural network...")
    nn_model, nn_scaler = train_neural_network(X_train, y_train, X_test, y_test)
    nn_preds = predict_neural_network(nn_model, nn_scaler, X_test)
    nn_metrics = evaluate_model(y_test, nn_preds, address_test)

    print_bias_crash_test_table(xgb_metrics, nn_metrics)

    save_baseline_artifact(xgb_model, feature_columns)
    save_fair_artifact(nn_model, nn_scaler, feature_columns)
    save_metrics(xgb_metrics, nn_metrics)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
