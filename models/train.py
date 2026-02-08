"""Walk-forward cross-validation, hyperparameter tuning, and ensemble training."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

from config import MIN_TEST_YEAR

warnings.filterwarnings("ignore", category=UserWarning)


def walk_forward_split(
    matchups: pd.DataFrame,
    feature_names: list[str],
    min_test_year: int = MIN_TEST_YEAR,
) -> list[tuple]:
    """Generate walk-forward train/test splits.

    Train on all data up to year N, test on year N+1.
    Respects temporal ordering — never trains on future data.

    Args:
        matchups: Full matchup DataFrame with season column.
        feature_names: Feature columns to use.
        min_test_year: Earliest year to use as a test set.

    Returns:
        List of (X_train, X_test, y_train, y_test, test_year) tuples.
    """
    seasons = sorted(matchups["season"].unique())
    splits = []

    for test_year in seasons:
        if test_year < min_test_year:
            continue

        train_mask = matchups["season"] < test_year
        test_mask = matchups["season"] == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = matchups.loc[train_mask, feature_names].values
        X_test = matchups.loc[test_mask, feature_names].values
        y_train = matchups.loc[train_mask, "home_win"].values
        y_test = matchups.loc[test_mask, "home_win"].values

        splits.append((X_train, X_test, y_train, y_test, test_year))

    return splits


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Train a Logistic Regression model with StandardScaler.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Tuple of (model, scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

    grid = GridSearchCV(lr, param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_scaled, y_train)

    return grid.best_estimator_, scaler


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train an XGBoost classifier with tuned hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained XGBClassifier.
    """
    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.1],
        "reg_lambda": [1.0],
    }

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
    """Train a LightGBM classifier with tuned hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained LGBMClassifier.
    """
    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.1],
        "reg_lambda": [1.0],
    }

    lgbm = LGBMClassifier(
        objective="binary",
        random_state=42,
        verbose=-1,
    )

    grid = GridSearchCV(lgbm, param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def train_ensemble(
    matchups: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Train all models using walk-forward validation and compute ensemble weights.

    Args:
        matchups: Full matchup DataFrame.
        feature_names: Feature columns.

    Returns:
        Dictionary with:
        - models: dict of trained models (on full data)
        - scaler: StandardScaler for LogisticRegression
        - weights: dict of ensemble weights
        - cv_results: list of per-fold results
        - feature_names: feature column names
    """
    splits = walk_forward_split(matchups, feature_names)

    # Track per-model performance across folds
    model_scores = {"lr": [], "xgb": [], "lgbm": []}
    cv_results = []

    print(f"\n  Walk-forward CV with {len(splits)} folds:")
    print(f"  {'Year':<8} {'LR Acc':<10} {'XGB Acc':<10} {'LGBM Acc':<10} {'Ens Acc':<10}")
    print(f"  {'-'*48}")

    for X_train, X_test, y_train, y_test, test_year in splits:
        # Train all three models
        lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)
        lgbm_model = train_lightgbm(X_train, y_train)

        # Predict probabilities
        lr_proba = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]

        # Accuracy
        lr_acc = accuracy_score(y_test, (lr_proba > 0.5).astype(int))
        xgb_acc = accuracy_score(y_test, (xgb_proba > 0.5).astype(int))
        lgbm_acc = accuracy_score(y_test, (lgbm_proba > 0.5).astype(int))

        # Simple ensemble (equal weight for now)
        ens_proba = (lr_proba + xgb_proba + lgbm_proba) / 3
        ens_acc = accuracy_score(y_test, (ens_proba > 0.5).astype(int))

        model_scores["lr"].append(lr_acc)
        model_scores["xgb"].append(xgb_acc)
        model_scores["lgbm"].append(lgbm_acc)

        cv_results.append({
            "test_year": test_year,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "lr_acc": lr_acc,
            "xgb_acc": xgb_acc,
            "lgbm_acc": lgbm_acc,
            "ens_acc": ens_acc,
            "lr_logloss": log_loss(y_test, lr_proba),
            "xgb_logloss": log_loss(y_test, xgb_proba),
            "lgbm_logloss": log_loss(y_test, lgbm_proba),
        })

        print(f"  {test_year:<8} {lr_acc:<10.3f} {xgb_acc:<10.3f} {lgbm_acc:<10.3f} {ens_acc:<10.3f}")

    # Compute ensemble weights based on average log loss (lower = better = higher weight)
    avg_logloss = {
        "lr": np.mean([r["lr_logloss"] for r in cv_results]),
        "xgb": np.mean([r["xgb_logloss"] for r in cv_results]),
        "lgbm": np.mean([r["lgbm_logloss"] for r in cv_results]),
    }

    # Inverse log loss weighting
    inv_ll = {k: 1.0 / v for k, v in avg_logloss.items()}
    total = sum(inv_ll.values())
    weights = {k: v / total for k, v in inv_ll.items()}

    print(f"\n  Ensemble weights (from validation log loss):")
    print(f"    LR: {weights['lr']:.3f}  XGB: {weights['xgb']:.3f}  LGBM: {weights['lgbm']:.3f}")

    # Retrain on ALL data for final prediction
    print(f"\n  Retraining all models on full dataset ({len(matchups):,} games)...")
    X_all = matchups[feature_names].values
    y_all = matchups["home_win"].values

    final_lr, final_scaler = train_logistic_regression(X_all, y_all)
    final_xgb = train_xgboost(X_all, y_all)
    final_lgbm = train_lightgbm(X_all, y_all)

    return {
        "models": {"lr": final_lr, "xgb": final_xgb, "lgbm": final_lgbm},
        "scaler": final_scaler,
        "weights": weights,
        "cv_results": cv_results,
        "feature_names": feature_names,
    }
