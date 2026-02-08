"""Fetch play-by-play and schedule data via nfl_data_py."""

import pandas as pd
import nfl_data_py as nfl

from config import PBP_COLUMNS, YEARS
from utils.constants import normalize_team


def load_pbp_data(years: list[int] | None = None) -> pd.DataFrame:
    """Load play-by-play data for the given seasons.

    Args:
        years: List of seasons to load. Defaults to config.YEARS.

    Returns:
        DataFrame with selected PBP columns and normalized team names.
    """
    if years is None:
        years = YEARS

    print(f"Loading play-by-play data for {years[0]}-{years[-1]}...")
    pbp = nfl.import_pbp_data(years, downcast=True)

    # Keep only columns that exist in the data
    available_cols = [c for c in PBP_COLUMNS if c in pbp.columns]
    pbp = pbp[available_cols].copy()

    # Normalize team abbreviations
    for col in ["home_team", "away_team", "posteam", "defteam"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].map(lambda x: normalize_team(x) if pd.notna(x) else x)

    # Filter to real plays (exclude no-plays like timeouts, end of quarter)
    pbp = pbp[pbp["play_type"].isin([
        "pass", "run", "punt", "field_goal", "extra_point",
        "kickoff", "qb_kneel", "qb_spike", "no_play",
    ])].copy()

    print(f"  Loaded {len(pbp):,} plays across {pbp['season'].nunique()} seasons")
    return pbp


def load_schedule_data(years: list[int] | None = None) -> pd.DataFrame:
    """Load schedule/game-level data for the given seasons.

    Args:
        years: List of seasons to load. Defaults to config.YEARS.

    Returns:
        DataFrame with game metadata including spreads, results, game type.
    """
    if years is None:
        years = YEARS

    print(f"Loading schedule data for {years[0]}-{years[-1]}...")
    schedule = nfl.import_schedules(years)

    # Normalize team abbreviations
    for col in ["home_team", "away_team"]:
        if col in schedule.columns:
            schedule[col] = schedule[col].map(
                lambda x: normalize_team(x) if pd.notna(x) else x
            )

    # Parse game date
    if "gameday" in schedule.columns:
        schedule["game_date"] = pd.to_datetime(schedule["gameday"])
    elif "game_date" not in schedule.columns:
        schedule["game_date"] = pd.NaT

    # Compute rest days
    schedule = schedule.sort_values(["season", "game_date"]).reset_index(drop=True)

    # Select useful columns
    keep_cols = [
        "game_id", "season", "week", "game_type",
        "home_team", "away_team",
        "home_score", "away_score",
        "spread_line", "total_line",
        "game_date", "weekday", "div_game",
        "roof", "surface",
    ]
    available = [c for c in keep_cols if c in schedule.columns]
    schedule = schedule[available].copy()

    # Add home win indicator
    if "home_score" in schedule.columns and "away_score" in schedule.columns:
        schedule["home_win"] = (
            schedule["home_score"] > schedule["away_score"]
        ).astype(int)
        # Drop ties (rare, simplifies binary classification)
        schedule = schedule[schedule["home_score"] != schedule["away_score"]].copy()

    print(f"  Loaded {len(schedule):,} games across {schedule['season'].nunique()} seasons")
    return schedule


def compute_rest_days(schedule: pd.DataFrame) -> pd.DataFrame:
    """Add rest day columns for home and away teams.

    Args:
        schedule: Schedule dataframe with game_date, home_team, away_team.

    Returns:
        Schedule with home_rest_days and away_rest_days columns added.
    """
    if "game_date" not in schedule.columns:
        schedule["home_rest_days"] = 7
        schedule["away_rest_days"] = 7
        schedule["rest_advantage"] = 0
        return schedule

    # Build a lookup of each team's last game date
    games_long = pd.concat([
        schedule[["game_date", "home_team", "season"]].rename(
            columns={"home_team": "team"}
        ),
        schedule[["game_date", "away_team", "season"]].rename(
            columns={"away_team": "team"}
        ),
    ]).sort_values("game_date")

    games_long["prev_game_date"] = games_long.groupby("team")["game_date"].shift(1)
    games_long["rest_days"] = (
        games_long["game_date"] - games_long["prev_game_date"]
    ).dt.days

    # Create separate rest lookups
    rest_lookup = games_long.set_index(["game_date", "team"])["rest_days"]

    schedule = schedule.copy()
    schedule["home_rest_days"] = schedule.apply(
        lambda r: rest_lookup.get((r["game_date"], r["home_team"]), 7), axis=1
    )
    schedule["away_rest_days"] = schedule.apply(
        lambda r: rest_lookup.get((r["game_date"], r["away_team"]), 7), axis=1
    )

    # Cap rest days at 21 (bye weeks can be long)
    schedule["home_rest_days"] = schedule["home_rest_days"].clip(upper=21).fillna(7)
    schedule["away_rest_days"] = schedule["away_rest_days"].clip(upper=21).fillna(7)
    schedule["rest_advantage"] = schedule["home_rest_days"] - schedule["away_rest_days"]

    return schedule
