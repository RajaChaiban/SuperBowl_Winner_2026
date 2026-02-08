"""Predict a specific matchup using the trained ensemble."""

import numpy as np
import pandas as pd


def predict_matchup(
    matchup_vector: pd.DataFrame,
    ensemble: dict,
    home_team: str,
    away_team: str,
) -> dict:
    """Run a matchup vector through the trained ensemble.

    Args:
        matchup_vector: Single-row DataFrame with feature columns.
        ensemble: Dict from train_ensemble with models, scaler, weights.
        home_team: Home team abbreviation.
        away_team: Away team abbreviation.

    Returns:
        Dictionary with prediction results.
    """
    models = ensemble["models"]
    scaler = ensemble["scaler"]
    weights = ensemble["weights"]
    feature_names = ensemble["feature_names"]

    X = matchup_vector[feature_names].values

    # Per-model probabilities
    lr_proba = models["lr"].predict_proba(scaler.transform(X))[0, 1]
    xgb_proba = models["xgb"].predict_proba(X)[0, 1]
    lgbm_proba = models["lgbm"].predict_proba(X)[0, 1]

    # Weighted ensemble
    ensemble_proba = (
        weights["lr"] * lr_proba
        + weights["xgb"] * xgb_proba
        + weights["lgbm"] * lgbm_proba
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": ensemble_proba,
        "away_win_prob": 1 - ensemble_proba,
        "predicted_winner": home_team if ensemble_proba > 0.5 else away_team,
        "model_breakdown": {
            "logistic_regression": {"home_prob": lr_proba, "weight": weights["lr"]},
            "xgboost": {"home_prob": xgb_proba, "weight": weights["xgb"]},
            "lightgbm": {"home_prob": lgbm_proba, "weight": weights["lgbm"]},
        },
    }


def get_top_factors(
    matchup_vector: pd.DataFrame,
    ensemble: dict,
    top_n: int = 10,
) -> list[dict]:
    """Identify the top contributing features for the prediction.

    Uses XGBoost feature importance weighted by the matchup differential values.

    Args:
        matchup_vector: Single-row DataFrame.
        ensemble: Trained ensemble dict.
        top_n: Number of top factors to return.

    Returns:
        List of dicts with feature name, importance, and direction.
    """
    feature_names = ensemble["feature_names"]
    xgb_model = ensemble["models"]["xgb"]

    # Get XGBoost feature importances
    importances = xgb_model.feature_importances_

    # Get the actual feature values
    values = matchup_vector[feature_names].values[0]

    factors = []
    for i, (name, imp, val) in enumerate(zip(feature_names, importances, values)):
        if imp > 0:
            factors.append({
                "feature": _format_feature_name(name),
                "raw_feature": name,
                "importance": float(imp),
                "value": float(val),
                "favors": "home" if val > 0 else "away" if val < 0 else "neutral",
            })

    # Sort by importance
    factors.sort(key=lambda x: x["importance"], reverse=True)
    return factors[:top_n]


def _format_feature_name(name: str) -> str:
    """Convert internal feature names to readable labels."""
    # Remove common prefixes
    name = name.replace("diff_rolling_3g_", "3-Game Rolling ")
    name = name.replace("diff_rolling_5g_", "5-Game Rolling ")
    name = name.replace("diff_season_avg_", "Season Avg ")
    name = name.replace("diff_", "")

    # Clean up specific names
    replacements = {
        "off_epa_per_play": "Offensive EPA/Play",
        "def_epa_per_play": "Defensive EPA/Play",
        "off_success_rate": "Offensive Success Rate",
        "off_yards_per_play": "Off Yards/Play",
        "off_pass_epa_per_play": "Pass EPA/Play",
        "off_rush_epa_per_play": "Rush EPA/Play",
        "off_completion_pct": "Completion %",
        "off_cpoe": "CPOE",
        "off_third_down_conv_rate": "3rd Down Conv Rate",
        "off_turnover_rate": "Turnover Rate",
        "def_sack_rate": "Sack Rate",
        "def_turnover_forced_rate": "Takeaway Rate",
        "elo_rating": "Elo Rating",
        "elo_diff": "Elo Differential",
        "strength_of_schedule": "Strength of Schedule",
        "point_differential": "Point Differential",
        "home_elo": "Home Elo",
        "away_elo": "Away Elo",
        "is_playoff": "Playoff Game",
        "spread_line": "Vegas Spread",
        "rest_advantage": "Rest Advantage",
    }

    for key, val in replacements.items():
        if key in name:
            name = name.replace(key, val)

    return name.strip()


def print_prediction(result: dict, factors: list[dict]) -> None:
    """Print a formatted prediction report.

    Args:
        result: Prediction result dict from predict_matchup.
        factors: Top factors from get_top_factors.
    """
    home = result["home_team"]
    away = result["away_team"]
    winner = result["predicted_winner"]
    loser = away if winner == home else home

    print("\n" + "=" * 60)
    print("SUPER BOWL PREDICTION")
    print("=" * 60)

    print(f"\n  {home} (home) vs {away} (away)")
    print(f"\n  PREDICTED WINNER: {winner}")
    print(f"  Win Probability: {max(result['home_win_prob'], result['away_win_prob']):.1%}")

    print(f"\n  Win Probabilities:")
    print(f"    {home}: {result['home_win_prob']:.1%}")
    print(f"    {away}: {result['away_win_prob']:.1%}")

    print(f"\n  Per-Model Breakdown:")
    for model_name, details in result["model_breakdown"].items():
        home_p = details["home_prob"]
        pred = home if home_p > 0.5 else away
        conf = max(home_p, 1 - home_p)
        print(f"    {model_name:<25} {pred} ({conf:.1%})  [weight: {details['weight']:.3f}]")

    if factors:
        print(f"\n  Top Contributing Factors:")
        for i, f in enumerate(factors, 1):
            direction = f"-> {home}" if f["favors"] == "home" else f"-> {away}" if f["favors"] == "away" else "neutral"
            print(f"    {i:2d}. {f['feature']:<35} {direction:<12} (imp: {f['importance']:.3f})")

    print("\n" + "=" * 60)
