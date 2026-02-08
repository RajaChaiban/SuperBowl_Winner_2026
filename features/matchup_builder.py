"""Layer 3: Build differential matchup feature vectors for game prediction."""

import pandas as pd
import numpy as np

from config import TEAM_STAT_FEATURES, ROLLING_WINDOWS
from utils.constants import is_divisional_game


def _get_feature_columns(team_stats: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns from team stats."""
    exclude = {"game_id", "season", "week", "team"}
    return [
        c for c in team_stats.columns
        if c not in exclude and team_stats[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]


def build_matchup_features(
    team_stats: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Build home-vs-away differential feature vectors for each game.

    For each stat, computes home_value - away_value. This reduces feature space
    and improves learning for head-to-head prediction.

    Args:
        team_stats: Per-game team stats with rolling/Elo/SOS features.
        schedule: Schedule with home_team, away_team, home_win target.

    Returns:
        DataFrame with differential features, context features, and home_win target.
    """
    # Use the most informative features: rolling averages and advanced features
    # (not raw per-game stats which are noisy)
    feature_cols = _get_feature_columns(team_stats)

    # Prefer rolling/season_avg/elo/sos features over raw stats
    model_features = []
    for col in feature_cols:
        if any(col.startswith(p) for p in ("rolling_", "season_avg_", "elo_", "strength_")):
            model_features.append(col)

    # Also include raw elo_rating
    if "elo_rating" in feature_cols and "elo_rating" not in model_features:
        model_features.append("elo_rating")

    # If we have very few rolling features, fall back to raw stats too
    if len(model_features) < 10:
        for col in TEAM_STAT_FEATURES:
            if col in feature_cols and col not in model_features:
                model_features.append(col)

    print(f"  Using {len(model_features)} base features for matchup vectors")

    # Index team stats for fast lookup
    team_stats_indexed = team_stats.set_index(["game_id", "team"])

    matchup_rows = []

    for _, game in schedule.iterrows():
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        # Get team stat rows
        try:
            home_stats = team_stats_indexed.loc[(gid, home)]
            away_stats = team_stats_indexed.loc[(gid, away)]
        except KeyError:
            continue

        row = {
            "game_id": gid,
            "season": game["season"],
            "week": game["week"],
            "home_team": home,
            "away_team": away,
        }

        # Differential features
        for col in model_features:
            h_val = home_stats.get(col, 0) if isinstance(home_stats, pd.Series) else 0
            a_val = away_stats.get(col, 0) if isinstance(away_stats, pd.Series) else 0

            # Handle potential NaN
            h_val = 0 if pd.isna(h_val) else float(h_val)
            a_val = 0 if pd.isna(a_val) else float(a_val)

            row[f"diff_{col}"] = h_val - a_val

        # Also keep raw Elo values (useful for the model)
        if "elo_rating" in model_features:
            h_elo = home_stats.get("elo_rating", 1500) if isinstance(home_stats, pd.Series) else 1500
            a_elo = away_stats.get("elo_rating", 1500) if isinstance(away_stats, pd.Series) else 1500
            row["home_elo"] = float(h_elo) if not pd.isna(h_elo) else 1500.0
            row["away_elo"] = float(a_elo) if not pd.isna(a_elo) else 1500.0
            row["elo_diff"] = row["home_elo"] - row["away_elo"]

        # Context features from schedule
        row["is_playoff"] = int(game.get("game_type", "REG") != "REG") if pd.notna(game.get("game_type")) else 0
        row["is_divisional"] = int(is_divisional_game(home, away))
        row["spread_line"] = float(game.get("spread_line", 0)) if pd.notna(game.get("spread_line")) else 0.0

        # Rest days
        row["home_rest_days"] = float(game.get("home_rest_days", 7)) if pd.notna(game.get("home_rest_days")) else 7.0
        row["away_rest_days"] = float(game.get("away_rest_days", 7)) if pd.notna(game.get("away_rest_days")) else 7.0
        row["rest_advantage"] = row["home_rest_days"] - row["away_rest_days"]

        # Target
        if "home_win" in game.index:
            row["home_win"] = int(game["home_win"])

        matchup_rows.append(row)

    matchups = pd.DataFrame(matchup_rows)

    # Fill any remaining NaN
    numeric_cols = matchups.select_dtypes(include=[np.number]).columns
    matchups[numeric_cols] = matchups[numeric_cols].fillna(0)

    feature_count = len([c for c in matchups.columns if c.startswith("diff_") or c in [
        "home_elo", "away_elo", "elo_diff", "is_playoff", "is_divisional",
        "spread_line", "home_rest_days", "away_rest_days", "rest_advantage",
    ]])
    print(f"  Built {len(matchups):,} matchup vectors with {feature_count} features")
    return matchups


def get_feature_names(matchups: pd.DataFrame) -> list[str]:
    """Get the feature column names used for modeling.

    Args:
        matchups: Matchup dataframe from build_matchup_features.

    Returns:
        List of feature column names (excludes identifiers and target).
    """
    exclude = {"game_id", "season", "week", "home_team", "away_team", "home_win"}
    return [c for c in matchups.columns if c not in exclude]


def build_prediction_matchup(
    home_team: str,
    away_team: str,
    team_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    feature_names: list[str],
    is_playoff: bool = True,
) -> pd.DataFrame:
    """Build a single matchup vector for prediction (e.g., Super Bowl).

    Uses each team's most recent stats as their current form.

    Args:
        home_team: Home team abbreviation.
        away_team: Away team abbreviation.
        team_stats: Full team stats with rolling/Elo features.
        schedule: Schedule data.
        feature_names: Feature columns expected by the model.
        is_playoff: Whether this is a playoff game.

    Returns:
        Single-row DataFrame with the same features as training data.
    """
    # Get most recent stats for each team
    home_latest = (
        team_stats[team_stats["team"] == home_team]
        .sort_values(["season", "week"])
        .iloc[-1]
    )
    away_latest = (
        team_stats[team_stats["team"] == away_team]
        .sort_values(["season", "week"])
        .iloc[-1]
    )

    row = {}
    for feat in feature_names:
        if feat.startswith("diff_"):
            base_col = feat[5:]  # Remove "diff_" prefix
            h_val = float(home_latest.get(base_col, 0))
            a_val = float(away_latest.get(base_col, 0))
            h_val = 0.0 if pd.isna(h_val) else h_val
            a_val = 0.0 if pd.isna(a_val) else a_val
            row[feat] = h_val - a_val
        elif feat == "home_elo":
            row[feat] = float(home_latest.get("elo_rating", 1500))
        elif feat == "away_elo":
            row[feat] = float(away_latest.get("elo_rating", 1500))
        elif feat == "elo_diff":
            h_elo = float(home_latest.get("elo_rating", 1500))
            a_elo = float(away_latest.get("elo_rating", 1500))
            row[feat] = h_elo - a_elo
        elif feat == "is_playoff":
            row[feat] = int(is_playoff)
        elif feat == "is_divisional":
            row[feat] = int(is_divisional_game(home_team, away_team))
        elif feat == "spread_line":
            row[feat] = 0.0  # No line for prediction
        elif feat == "home_rest_days":
            row[feat] = 14.0  # Super Bowl typical rest
        elif feat == "away_rest_days":
            row[feat] = 14.0
        elif feat == "rest_advantage":
            row[feat] = 0.0
        else:
            row[feat] = 0.0

    matchup = pd.DataFrame([row])

    # Ensure NaN-free
    matchup = matchup.fillna(0)
    return matchup
