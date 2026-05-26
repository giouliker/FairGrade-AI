"""
Data preparation for the FairGrade AI tabular pipeline.

This module loads the public UCI Student Performance (Mathematics) dataset,
engineers numeric interaction features, one-hot encodes categorical fields,
and exposes ``address`` (Urban/Rural) only as offline fairness metadata.

No personal identifiers, credentials, or machine-specific paths are used.
All file locations are relative ``pathlib.Path`` objects resolved from the
project working directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import DATA_PATH

# Re-export for callers that import DATA_PATH from this module.
__all__ = ["DATA_PATH", "load_and_preprocess_data", "build_feature_matrix", "align_to_training_columns"]

# Raw numeric columns passed directly to the model after engineering.
NUMERIC_FEATURES = ["studytime", "failures", "absences", "goout", "Medu", "Fedu"]

# Categorical columns expanded with pandas.get_dummies (see build_feature_matrix).
CATEGORICAL_FEATURES = ["Mjob", "Fjob", "reason"]

# Columns created in engineer_features() and appended to the numeric block.
ENGINEERED_FEATURES = ["parent_edu_total", "study_vs_out"]

# Final grade threshold for binary pass/fail (UCI scale 0–20).
PASS_GRADE_THRESHOLD = 10


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """
    Load the UCI Student Performance CSV into a pandas DataFrame.

    Parameters
    ----------
    path:
        Relative or absolute path to ``student-mat.csv``. Defaults to
        ``data/student-mat.csv`` under the project root.

    Returns
    -------
    pd.DataFrame
        Raw records with original column names (e.g. ``G3``, ``address``).
    """
    csv_path = DATA_PATH if path is None else path
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Ensure data/student-mat.csv is committed to the repository."
        )
    return pd.read_csv(csv_path, sep=";")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features that capture family context and study habits.

    Features
    --------
    parent_edu_total:
        Sum of mother's and father's education codes (``Medu`` + ``Fedu``).
        Higher values indicate more combined parental schooling.
    study_vs_out:
        Ratio ``studytime / (goout + 1)``. The ``+ 1`` on ``goout`` avoids
        division by zero when a student rarely goes out. This ratio proxies
        whether study time dominates social time.

    Parameters
    ----------
    df:
        DataFrame containing the base numeric and categorical columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the two engineered columns appended.
    """
    out = df.copy()
    out["parent_edu_total"] = out["Medu"] + out["Fedu"]
    out["study_vs_out"] = out["studytime"] / (out["goout"] + 1)
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model-ready design matrix ``X`` from raw student rows.

    Pipeline
    --------
    1. Run :func:`engineer_features` on the input frame.
    2. Select base numerics plus engineered numerics.
    3. One-hot encode ``Mjob``, ``Fjob``, and ``reason``.
    4. Concatenate and sort column names for a stable schema across runs.

    One-hot encoding (``drop_first=True``)
    --------------------------------------
    For each categorical column, pandas creates dummy variables for all
    levels except one reference category. That reference is implied when
    all dummies for that column are zero. We use ``drop_first=True`` to:

    - Avoid the **dummy variable trap** (perfect multicollinearity among
      dummies for the same field, which harms linear models and can
      destabilize neural nets).
    - Reduce dimensionality slightly on this small tabular dataset.

    The exact dummy columns depend on pandas' category ordering; training
    and inference must share the same column list (saved in model artifacts
    and enforced via :func:`align_to_training_columns` at inference time).

    Parameters
    ----------
    df:
        Raw or partially processed student records.

    Returns
    -------
    pd.DataFrame
        Numeric + one-hot feature matrix. Does **not** include ``address``.
    """
    engineered = engineer_features(df)

    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    X_numeric = engineered[numeric_cols]

    # drop_first=True: one baseline category per categorical field is omitted.
    X_categorical = pd.get_dummies(engineered[CATEGORICAL_FEATURES], drop_first=True)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    # Sorted columns ensure train.py and app.py produce identical ordering.
    return X.reindex(sorted(X.columns), axis=1)


def align_to_training_columns(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Align an inference-time feature matrix to the training-time column schema.

    At prediction time, a single sidebar profile may not activate every dummy
    column seen during training (e.g. a rare job category). ``reindex`` adds
    missing columns filled with ``0`` and drops any extras, so both XGBoost
    and the neural network always receive a vector of the expected shape.

    Parameters
    ----------
    X:
        Feature matrix from :func:`build_feature_matrix` for one or more rows.
    feature_columns:
        Column names persisted in ``baseline_model.pkl`` / ``fair_model.pt``.

    Returns
    -------
    pd.DataFrame
        ``X`` with exactly ``feature_columns`` in that order.
    """
    return X.reindex(columns=feature_columns, fill_value=0)


def load_and_preprocess_data(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    End-to-end load: features, binary target, and fairness metadata.

    Target definition
    -----------------
    ``passed = 1`` if final grade ``G3 >= PASS_GRADE_THRESHOLD`` (10/20),
    else ``0``. This matches a simple pass/fail decision rule on the UCI scale.

    Fairness metadata
    -----------------
    ``address`` is extracted as ``'U'`` (Urban) or ``'R'`` (Rural) but is
    **never** included in ``X``. It is used only in ``train.py`` to compute
    disparate impact on the hold-out set.

    Parameters
    ----------
    path:
        Path to the student CSV file.

    Returns
    -------
    X : pd.DataFrame
        Full training feature matrix.
    y : pd.Series
        Binary pass/fail labels.
    address : pd.Series
        Urban/Rural codes for fairness evaluation only.
    """
    df = load_raw_dataframe(path)

    df["passed"] = (df["G3"] >= PASS_GRADE_THRESHOLD).astype(int)

    # Held out of X by design — not a model input.
    address = df["address"].reset_index(drop=True)

    X = build_feature_matrix(df)
    y = df["passed"].reset_index(drop=True)

    print("--- Data preparation complete ---")
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Pass rate: {y.mean():.1%}")
    print(f"Address distribution:\n{address.value_counts().to_string()}\n")

    return X, y, address


if __name__ == "__main__":
    # Quick sanity check when run as a script from the project root.
    features, labels, addresses = load_and_preprocess_data()
    print(f"Feature columns ({features.shape[1]}): {list(features.columns)}")
