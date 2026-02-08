"""NFL Super Bowl Predictor — Full Pipeline Orchestrator.

Predicts the Super Bowl winner between Seattle Seahawks and New England Patriots
using historical NFL data (2015-2024) with an ensemble of ML models.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Pipeline imports
from data.loader import load_pbp_data, load_schedule_data, compute_rest_days
from features.team_stats import compute_team_game_stats, add_game_level_stats
from features.advanced_features import (
    add_rolling_averages,
    compute_elo_ratings,
    add_elo_to_team_stats,
    compute_strength_of_schedule,
)
from features.matchup_builder import (
    build_matchup_features,
    get_feature_names,
    build_prediction_matchup,
)
from models.train import train_ensemble
from models.evaluate import print_evaluation_report
from models.predict import predict_matchup, get_top_factors, print_prediction


# === CONFIGURATION ===
HOME_TEAM = "NE"    # New England Patriots
AWAY_TEAM = "SEA"   # Seattle Seahawks


def main():
    """Run the full NFL prediction pipeline."""
    start_time = time.time()

    print("=" * 60)
    print("NFL SUPER BOWL PREDICTOR")
    print(f"{AWAY_TEAM} @ {HOME_TEAM}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 1: DATA LOADING
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Loading Data...")
    pbp = load_pbp_data()
    schedule = load_schedule_data()
    schedule = compute_rest_days(schedule)

    # ------------------------------------------------------------------
    # PHASE 2: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    print("\n[PHASE 2] Engineering Features...")

    # Layer 1: Per-game team stats from play-by-play
    print("\n  Layer 1: Computing per-game team statistics...")
    team_stats = compute_team_game_stats(pbp)
    team_stats = add_game_level_stats(team_stats, schedule)

    # Layer 2: Temporal features
    print("\n  Layer 2: Adding temporal features...")
    elo_df = compute_elo_ratings(schedule)
    team_stats = add_elo_to_team_stats(team_stats, elo_df)
    team_stats = compute_strength_of_schedule(team_stats, schedule, elo_df)
    team_stats = add_rolling_averages(team_stats)

    # Layer 3: Matchup differential vectors
    print("\n  Layer 3: Building matchup feature vectors...")
    matchups = build_matchup_features(team_stats, schedule)
    feature_names = get_feature_names(matchups)

    print(f"\n  Dataset summary:")
    print(f"    Total matchups: {len(matchups):,}")
    print(f"    Features per matchup: {len(feature_names)}")
    print(f"    Seasons: {matchups['season'].min()}-{matchups['season'].max()}")
    print(f"    Home win rate: {matchups['home_win'].mean():.1%}")

    # ------------------------------------------------------------------
    # PHASE 3: MODEL TRAINING
    # ------------------------------------------------------------------
    print("\n[PHASE 3] Training Models...")
    ensemble = train_ensemble(matchups, feature_names)

    # ------------------------------------------------------------------
    # PHASE 4: EVALUATION
    # ------------------------------------------------------------------
    print("\n[PHASE 4] Evaluation...")
    print_evaluation_report(ensemble["cv_results"], ensemble["weights"])

    # ------------------------------------------------------------------
    # PHASE 5: SUPER BOWL PREDICTION
    # ------------------------------------------------------------------
    print("\n[PHASE 5] Predicting Super Bowl...")

    # Verify both teams exist in data
    teams_in_data = team_stats["team"].unique()
    for team in [HOME_TEAM, AWAY_TEAM]:
        if team not in teams_in_data:
            print(f"  ERROR: {team} not found in data. Available: {sorted(teams_in_data)}")
            sys.exit(1)

    # Build prediction matchup vector
    matchup_vector = build_prediction_matchup(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        team_stats=team_stats,
        schedule=schedule,
        feature_names=feature_names,
        is_playoff=True,
    )

    # Predict
    result = predict_matchup(matchup_vector, ensemble, HOME_TEAM, AWAY_TEAM)
    factors = get_top_factors(matchup_vector, ensemble, top_n=10)

    # Display results
    print_prediction(result, factors)

    # Team snapshots
    print("\nTEAM SNAPSHOTS (Latest Available Stats):")
    for team in [HOME_TEAM, AWAY_TEAM]:
        latest = (
            team_stats[team_stats["team"] == team]
            .sort_values(["season", "week"])
            .iloc[-1]
        )
        elo = latest.get("elo_rating", 1500)
        sos = latest.get("strength_of_schedule", 1500)
        off_epa = latest.get("season_avg_off_epa_per_play", latest.get("off_epa_per_play", 0))
        def_epa = latest.get("season_avg_def_epa_per_play", latest.get("def_epa_per_play", 0))

        print(f"\n  {team}:")
        print(f"    Elo Rating:    {elo:.0f}")
        print(f"    SOS:           {sos:.0f}")
        print(f"    Off EPA/Play:  {off_epa:.3f}")
        print(f"    Def EPA/Play:  {def_epa:.3f}")

    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
