"""Model evaluation: accuracy, log loss, Brier score, AUC-ROC, calibration."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)


def evaluate_cv_results(cv_results: list[dict]) -> dict:
    """Compute aggregate metrics from walk-forward CV results.

    Args:
        cv_results: List of per-fold result dicts from train_ensemble.

    Returns:
        Dictionary of aggregate metrics.
    """
    metrics = {}

    for model_key in ["lr", "xgb", "lgbm"]:
        acc_key = f"{model_key}_acc"
        ll_key = f"{model_key}_logloss"

        accs = [r[acc_key] for r in cv_results]
        lls = [r[ll_key] for r in cv_results]

        metrics[model_key] = {
            "mean_accuracy": np.mean(accs),
            "std_accuracy": np.std(accs),
            "mean_logloss": np.mean(lls),
            "std_logloss": np.std(lls),
        }

    # Ensemble accuracy
    ens_accs = [r["ens_acc"] for r in cv_results]
    metrics["ensemble"] = {
        "mean_accuracy": np.mean(ens_accs),
        "std_accuracy": np.std(ens_accs),
    }

    return metrics


def print_evaluation_report(cv_results: list[dict], weights: dict) -> None:
    """Print a formatted evaluation report.

    Args:
        cv_results: Walk-forward CV results.
        weights: Ensemble model weights.
    """
    metrics = evaluate_cv_results(cv_results)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Accuracy':<18} {'Log Loss':<18}")
    print(f"{'-'*56}")

    for name, label in [("lr", "Logistic Reg"), ("xgb", "XGBoost"), ("lgbm", "LightGBM")]:
        m = metrics[name]
        print(
            f"{label:<20} "
            f"{m['mean_accuracy']:.3f} +/- {m['std_accuracy']:.3f}   "
            f"{m['mean_logloss']:.3f} +/- {m['std_logloss']:.3f}"
        )

    m = metrics["ensemble"]
    print(f"{'Ensemble':<20} {m['mean_accuracy']:.3f} +/- {m['std_accuracy']:.3f}")

    print(f"\nEnsemble Weights:")
    print(f"  LR: {weights['lr']:.3f}  XGB: {weights['xgb']:.3f}  LGBM: {weights['lgbm']:.3f}")

    # Baseline comparison
    baseline = 0.57  # Home team win rate
    best_acc = metrics["ensemble"]["mean_accuracy"]
    improvement = best_acc - baseline
    print(f"\nBaseline (home wins): {baseline:.1%}")
    print(f"Ensemble improvement: +{improvement:.1%}")

    # Per-year breakdown
    print(f"\nPer-Year Breakdown:")
    print(f"{'Year':<8} {'Games':<8} {'LR':<8} {'XGB':<8} {'LGBM':<8} {'Ens':<8}")
    print(f"{'-'*48}")
    for r in cv_results:
        print(
            f"{r['test_year']:<8} {r['n_test']:<8} "
            f"{r['lr_acc']:<8.3f} {r['xgb_acc']:<8.3f} "
            f"{r['lgbm_acc']:<8.3f} {r['ens_acc']:<8.3f}"
        )


def compute_prediction_confidence(probabilities: dict[str, float]) -> str:
    """Classify prediction confidence level.

    Args:
        probabilities: Dict mapping team names to win probabilities.

    Returns:
        Confidence label string.
    """
    max_prob = max(probabilities.values())

    if max_prob >= 0.70:
        return "HIGH"
    elif max_prob >= 0.60:
        return "MODERATE"
    elif max_prob >= 0.55:
        return "SLIGHT"
    else:
        return "TOSS-UP"
