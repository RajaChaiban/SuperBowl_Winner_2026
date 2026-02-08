"""Layer 2: Temporal features — rolling averages, Elo ratings, strength of schedule."""

import pandas as pd
import numpy as np

from config import (
    TEAM_STAT_FEATURES,
    ROLLING_WINDOWS,
    ELO_K,
    ELO_HOME_ADVANTAGE,
    ELO_MEAN,
    ELO_SEASON_REVERSION,
)


def add_rolling_averages(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and expanding window averages for team stats.

    Uses .shift(1) to prevent data leakage — only prior-game data.

    Args:
        team_stats: Per-game team stats sorted by (team, season, week).

    Returns:
        DataFrame with rolling_{w}g_ and season_avg_ columns added.
    """
    team_stats = team_stats.sort_values(["team", "season", "week"]).copy()

    # Identify numeric feature columns present in data
    stat_cols = [c for c in TEAM_STAT_FEATURES if c in team_stats.columns]

    for col in stat_cols:
        grouped = team_stats.groupby("team")[col]

        # Rolling windows (shift to prevent leakage)
        for w in ROLLING_WINDOWS:
            team_stats[f"rolling_{w}g_{col}"] = (
                grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        # Season-to-date expanding average (shift to prevent leakage)
        team_stats[f"season_avg_{col}"] = (
            team_stats.groupby(["team", "season"])[col]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

    # Fill NaN for early-season games (no prior data)
    rolling_cols = [c for c in team_stats.columns if c.startswith(("rolling_", "season_avg_"))]
    team_stats[rolling_cols] = team_stats[rolling_cols].fillna(0)

    print(f"  Added {len(rolling_cols)} rolling/expanding features")
    return team_stats


def compute_elo_ratings(schedule: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for all teams across seasons.

    Args:
        schedule: Game-level data with home_team, away_team, home_score, away_score.

    Returns:
        DataFrame with columns: game_id, team, elo_before, elo_after.
    """
    schedule = schedule.sort_values(["season", "game_date", "week"]).copy()

    elo = {}  # Current Elo for each team
    records = []

    prev_season = None

    for _, game in schedule.iterrows():
        season = game["season"]
        home = game["home_team"]
        away = game["away_team"]
        gid = game["game_id"]

        # Season reversion toward mean
        if season != prev_season:
            for team in list(elo.keys()):
                elo[team] = elo[team] + ELO_SEASON_REVERSION * (ELO_MEAN - elo[team])
            prev_season = season

        # Initialize new teams
        if home not in elo:
            elo[home] = ELO_MEAN
        if away not in elo:
            elo[away] = ELO_MEAN

        home_elo = elo[home]
        away_elo = elo[away]

        # Expected win probability (with home advantage)
        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400))
        exp_away = 1.0 - exp_home

        # Actual result
        home_score = game.get("home_score", 0) or 0
        away_score = game.get("away_score", 0) or 0

        if home_score > away_score:
            actual_home = 1.0
        elif away_score > home_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Margin of victory multiplier
        mov = abs(home_score - away_score)
        mov_mult = np.log(max(mov, 1) + 1) * (2.2 / (2.2 + 0.001 * abs(home_elo - away_elo)))

        # Update Elo
        shift = ELO_K * mov_mult * (actual_home - exp_home)
        new_home_elo = home_elo + shift
        new_away_elo = away_elo - shift

        records.append({
            "game_id": gid,
            "team": home,
            "elo_before": home_elo,
            "elo_after": new_home_elo,
        })
        records.append({
            "game_id": gid,
            "team": away,
            "elo_before": away_elo,
            "elo_after": new_away_elo,
        })

        elo[home] = new_home_elo
        elo[away] = new_away_elo

    elo_df = pd.DataFrame(records)
    print(f"  Computed Elo ratings for {elo_df['team'].nunique()} teams, "
          f"{elo_df['game_id'].nunique()} games")
    return elo_df


def add_elo_to_team_stats(
    team_stats: pd.DataFrame, elo_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge Elo ratings into team stats.

    Uses elo_before (pre-game Elo) to avoid leakage.

    Args:
        team_stats: Per-game team stats.
        elo_df: Elo ratings from compute_elo_ratings.

    Returns:
        team_stats with elo_rating column added.
    """
    elo_merge = elo_df[["game_id", "team", "elo_before"]].rename(
        columns={"elo_before": "elo_rating"}
    )
    team_stats = team_stats.merge(elo_merge, on=["game_id", "team"], how="left")
    team_stats["elo_rating"] = team_stats["elo_rating"].fillna(ELO_MEAN)
    return team_stats


def compute_strength_of_schedule(
    team_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling strength of schedule (avg opponent Elo).

    Args:
        team_stats: Per-game team stats with elo_rating.
        schedule: Schedule data.
        elo_df: Elo ratings.

    Returns:
        team_stats with strength_of_schedule column.
    """
    # Build opponent mapping from schedule
    opponent_map = {}
    for _, row in schedule.iterrows():
        gid = row["game_id"]
        opponent_map[(gid, row["home_team"])] = row["away_team"]
        opponent_map[(gid, row["away_team"])] = row["home_team"]

    # Map opponent Elo
    elo_lookup = elo_df.set_index(["game_id", "team"])["elo_before"].to_dict()

    def get_opponent_elo(row):
        opp = opponent_map.get((row["game_id"], row["team"]))
        if opp is None:
            return ELO_MEAN
        return elo_lookup.get((row["game_id"], opp), ELO_MEAN)

    team_stats = team_stats.copy()
    team_stats["_opp_elo"] = team_stats.apply(get_opponent_elo, axis=1)

    # Rolling average of opponent Elo (shift to prevent leakage)
    team_stats = team_stats.sort_values(["team", "season", "week"])
    team_stats["strength_of_schedule"] = (
        team_stats.groupby("team")["_opp_elo"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    team_stats["strength_of_schedule"] = team_stats["strength_of_schedule"].fillna(ELO_MEAN)
    team_stats = team_stats.drop(columns=["_opp_elo"])

    print(f"  Added strength of schedule feature")
    return team_stats
