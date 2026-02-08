"""Layer 1: Aggregate play-by-play data into per-game team statistics."""

import pandas as pd
import numpy as np


def _safe_div(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, filling division-by-zero with fill value."""
    return numerator.div(denominator).replace([np.inf, -np.inf], fill).fillna(fill)


def compute_team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game, per-team offensive and defensive statistics.

    Args:
        pbp: Play-by-play dataframe with EPA, yards, plays.

    Returns:
        DataFrame indexed by (game_id, season, week, team) with ~33 features.
    """
    # Only real scrimmage plays for offensive/defensive stats
    scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # --- Offensive stats (when team is possessing) ---
    off_groups = scrimmage.groupby(["game_id", "season", "week", "posteam"])

    off = pd.DataFrame()
    off["off_epa_per_play"] = off_groups["epa"].mean()
    off["off_total_epa"] = off_groups["epa"].sum()
    off["off_success_rate"] = off_groups["success"].mean()
    off["off_yards_per_play"] = off_groups["yards_gained"].mean()
    off["off_first_down_rate"] = off_groups["first_down"].mean()
    off["off_total_plays"] = off_groups["epa"].count()

    # Passing
    pass_plays = scrimmage[scrimmage["play_type"] == "pass"]
    pass_groups = pass_plays.groupby(["game_id", "season", "week", "posteam"])

    pass_stats = pd.DataFrame()
    pass_stats["_pass_epa_sum"] = pass_groups["epa"].sum()
    pass_stats["_pass_attempts"] = pass_groups["pass_attempt"].sum()
    pass_stats["_completions"] = pass_groups["complete_pass"].sum()
    pass_stats["_pass_yards"] = pass_groups["yards_gained"].sum()
    pass_stats["_interceptions_thrown"] = pass_groups["interception"].sum()
    pass_stats["_pass_tds"] = pass_groups["pass_touchdown"].sum()
    pass_stats["_sacks_taken"] = pass_groups["sack"].sum()
    pass_stats["_cpoe_mean"] = pass_groups["cpoe"].mean()

    off = off.join(pass_stats, how="left")
    off["off_pass_epa_per_play"] = _safe_div(off["_pass_epa_sum"], off["_pass_attempts"])
    off["off_pass_yards_per_att"] = _safe_div(off["_pass_yards"], off["_pass_attempts"])
    off["off_completion_pct"] = _safe_div(off["_completions"], off["_pass_attempts"])
    off["off_cpoe"] = off["_cpoe_mean"].fillna(0)
    off["off_pass_td_rate"] = _safe_div(off["_pass_tds"], off["_pass_attempts"])
    off["off_sack_rate_allowed"] = _safe_div(
        off["_sacks_taken"], off["_sacks_taken"] + off["_pass_attempts"]
    )

    # Rushing
    rush_plays = scrimmage[scrimmage["play_type"] == "run"]
    rush_groups = rush_plays.groupby(["game_id", "season", "week", "posteam"])

    rush_stats = pd.DataFrame()
    rush_stats["_rush_epa_sum"] = rush_groups["epa"].sum()
    rush_stats["_rush_attempts"] = rush_groups["rush_attempt"].sum()
    rush_stats["_rush_yards"] = rush_groups["yards_gained"].sum()
    rush_stats["_rush_tds"] = rush_groups["rush_touchdown"].sum()

    off = off.join(rush_stats, how="left")
    off["off_rush_epa_per_play"] = _safe_div(off["_rush_epa_sum"], off["_rush_attempts"])
    off["off_rush_yards_per_att"] = _safe_div(off["_rush_yards"], off["_rush_attempts"])
    off["off_rush_td_rate"] = _safe_div(off["_rush_tds"], off["_rush_attempts"])

    # Third down
    third_down = scrimmage[scrimmage["down"] == 3]
    td_groups = third_down.groupby(["game_id", "season", "week", "posteam"])
    td_stats = pd.DataFrame()
    td_stats["_3d_conv"] = td_groups["third_down_converted"].sum()
    td_stats["_3d_att"] = td_groups["third_down_converted"].count()

    off = off.join(td_stats, how="left")
    off["off_third_down_conv_rate"] = _safe_div(off["_3d_conv"], off["_3d_att"])

    # Turnovers committed
    off["_turnovers_committed"] = off["_interceptions_thrown"].fillna(0)
    fumbles = scrimmage.groupby(["game_id", "season", "week", "posteam"])["fumble_lost"].sum()
    off["_turnovers_committed"] = off["_turnovers_committed"] + fumbles.reindex(off.index).fillna(0)
    off["off_turnover_rate"] = _safe_div(off["_turnovers_committed"], off["off_total_plays"])

    # Penalties
    pen_groups = scrimmage.groupby(["game_id", "season", "week", "posteam"])
    off["off_penalty_rate"] = pen_groups["penalty"].mean()

    # No-huddle rate
    off["off_no_huddle_rate"] = off_groups["no_huddle"].mean()

    # Scoring rate (TDs per play)
    off["off_scoring_rate"] = _safe_div(
        off["_pass_tds"].fillna(0) + off["_rush_tds"].fillna(0),
        off["off_total_plays"],
    )

    # --- Defensive stats (when team is defending) ---
    def_groups = scrimmage.groupby(["game_id", "season", "week", "defteam"])

    defense = pd.DataFrame()
    defense["def_epa_per_play"] = def_groups["epa"].mean()
    defense["def_success_rate_allowed"] = def_groups["success"].mean()
    defense["def_yards_per_play_allowed"] = def_groups["yards_gained"].mean()
    defense["def_total_plays_faced"] = def_groups["epa"].count()

    # Defensive pass stats
    def_pass = pass_plays.groupby(["game_id", "season", "week", "defteam"])
    def_pass_stats = pd.DataFrame()
    def_pass_stats["_def_pass_epa_sum"] = def_pass["epa"].sum()
    def_pass_stats["_def_pass_att_faced"] = def_pass["pass_attempt"].sum()
    def_pass_stats["_def_completions_allowed"] = def_pass["complete_pass"].sum()
    def_pass_stats["_def_ints_forced"] = def_pass["interception"].sum()
    def_pass_stats["_def_sacks"] = def_pass["sack"].sum()
    def_pass_stats["_def_pass_tds_allowed"] = def_pass["pass_touchdown"].sum()

    defense = defense.join(def_pass_stats, how="left")
    defense["def_pass_epa_per_play"] = _safe_div(
        defense["_def_pass_epa_sum"], defense["_def_pass_att_faced"]
    )
    defense["def_completion_pct_allowed"] = _safe_div(
        defense["_def_completions_allowed"], defense["_def_pass_att_faced"]
    )
    defense["def_sack_rate"] = _safe_div(
        defense["_def_sacks"],
        defense["_def_sacks"] + defense["_def_pass_att_faced"],
    )
    defense["def_int_rate"] = _safe_div(
        defense["_def_ints_forced"], defense["_def_pass_att_faced"]
    )

    # Defensive rush stats
    def_rush = rush_plays.groupby(["game_id", "season", "week", "defteam"])
    def_rush_stats = pd.DataFrame()
    def_rush_stats["_def_rush_epa_sum"] = def_rush["epa"].sum()
    def_rush_stats["_def_rush_att_faced"] = def_rush["rush_attempt"].sum()

    defense = defense.join(def_rush_stats, how="left")
    defense["def_rush_epa_per_play"] = _safe_div(
        defense["_def_rush_epa_sum"], defense["_def_rush_att_faced"]
    )

    # Third down defense
    def_3d = third_down.groupby(["game_id", "season", "week", "defteam"])
    def_3d_stats = pd.DataFrame()
    def_3d_stats["_def_3d_conv"] = def_3d["third_down_converted"].sum()
    def_3d_stats["_def_3d_att"] = def_3d["third_down_converted"].count()

    defense = defense.join(def_3d_stats, how="left")
    defense["def_third_down_conv_rate_allowed"] = _safe_div(
        defense["_def_3d_conv"], defense["_def_3d_att"]
    )

    # Turnovers forced
    def_fumbles = scrimmage.groupby(["game_id", "season", "week", "defteam"])["fumble_lost"].sum()
    defense["_turnovers_forced"] = (
        defense["_def_ints_forced"].fillna(0) + def_fumbles.reindex(defense.index).fillna(0)
    )
    defense["def_turnover_forced_rate"] = _safe_div(
        defense["_turnovers_forced"], defense["def_total_plays_faced"]
    )

    # Defensive penalties
    defense["def_penalty_rate"] = def_groups["penalty"].mean()

    # Scoring rate allowed
    def_rush_tds = rush_plays.groupby(["game_id", "season", "week", "defteam"])[
        "rush_touchdown"
    ].sum()
    defense["def_scoring_rate_allowed"] = _safe_div(
        defense["_def_pass_tds_allowed"].fillna(0)
        + def_rush_tds.reindex(defense.index).fillna(0),
        defense["def_total_plays_faced"],
    )

    # --- Special teams ---
    fg_plays = pbp[pbp["play_type"] == "field_goal"]
    fg_groups = fg_plays.groupby(["game_id", "season", "week", "posteam"])
    fg_stats = pd.DataFrame()
    if len(fg_plays) > 0:
        fg_stats["_fg_made"] = fg_groups["field_goal_result"].apply(
            lambda x: (x == "made").sum()
        )
        fg_stats["_fg_att"] = fg_groups["field_goal_attempt"].sum()

    xp_plays = pbp[pbp["play_type"] == "extra_point"]
    xp_groups = xp_plays.groupby(["game_id", "season", "week", "posteam"])
    xp_stats = pd.DataFrame()
    if len(xp_plays) > 0:
        xp_stats["_xp_made"] = xp_groups["extra_point_result"].apply(
            lambda x: (x == "good").sum()
        )
        xp_stats["_xp_att"] = xp_groups["extra_point_attempt"].sum()

    punt_plays = pbp[pbp["play_type"] == "punt"]
    punt_groups = punt_plays.groupby(["game_id", "season", "week", "posteam"])
    punt_stats = pd.DataFrame()
    if len(punt_plays) > 0:
        punt_stats["_punts"] = punt_groups["punt_attempt"].sum()

    # --- Combine all into per-game team stats ---
    # Rename index levels for joining
    off.index.names = ["game_id", "season", "week", "team"]
    defense.index.names = ["game_id", "season", "week", "team"]

    # Join offensive and defensive
    team_stats = off.join(defense, how="outer")

    # Join special teams
    for st in [fg_stats, xp_stats, punt_stats]:
        if len(st) > 0:
            st.index.names = ["game_id", "season", "week", "team"]
            team_stats = team_stats.join(st, how="left")

    # Compute special teams features
    team_stats["st_fg_pct"] = _safe_div(
        team_stats.get("_fg_made", pd.Series(0, index=team_stats.index)),
        team_stats.get("_fg_att", pd.Series(1, index=team_stats.index)),
    )
    team_stats["st_xp_pct"] = _safe_div(
        team_stats.get("_xp_made", pd.Series(0, index=team_stats.index)),
        team_stats.get("_xp_att", pd.Series(1, index=team_stats.index)),
    )
    total_plays = team_stats["off_total_plays"].fillna(0)
    team_stats["st_punt_rate"] = _safe_div(
        team_stats.get("_punts", pd.Series(0, index=team_stats.index)),
        total_plays,
    )

    # Drop intermediate columns
    intermediate = [c for c in team_stats.columns if c.startswith("_")]
    team_stats = team_stats.drop(columns=intermediate, errors="ignore")

    # Reset index
    team_stats = team_stats.reset_index()

    print(f"  Computed {len(team_stats.columns) - 4} features for "
          f"{len(team_stats):,} team-games")
    return team_stats


def add_game_level_stats(
    team_stats: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Add game-level features like point differential and turnover differential.

    Args:
        team_stats: Per-game team stats from compute_team_game_stats.
        schedule: Schedule with home_score, away_score.

    Returns:
        team_stats with point_differential, turnover_differential, total_plays added.
    """
    # Build score lookup from schedule
    if "home_score" not in schedule.columns:
        team_stats["point_differential"] = 0
        team_stats["turnover_differential"] = 0
        team_stats["total_plays"] = team_stats.get("off_total_plays", 60)
        return team_stats

    score_lookup = {}
    for _, row in schedule.iterrows():
        gid = row["game_id"]
        score_lookup[(gid, row["home_team"])] = (
            row.get("home_score", 0),
            row.get("away_score", 0),
        )
        score_lookup[(gid, row["away_team"])] = (
            row.get("away_score", 0),
            row.get("home_score", 0),
        )

    def get_point_diff(row):
        key = (row["game_id"], row["team"])
        scores = score_lookup.get(key, (0, 0))
        return scores[0] - scores[1]

    team_stats["point_differential"] = team_stats.apply(get_point_diff, axis=1)

    # Turnover differential = turnovers_forced - turnovers_committed
    team_stats["turnover_differential"] = (
        team_stats["def_turnover_forced_rate"].fillna(0)
        - team_stats["off_turnover_rate"].fillna(0)
    ) * team_stats.get("off_total_plays", 60)

    # Total plays
    team_stats["total_plays"] = (
        team_stats["off_total_plays"].fillna(0)
        + team_stats["def_total_plays_faced"].fillna(0)
    )

    # Drop raw count columns
    team_stats = team_stats.drop(
        columns=["off_total_plays", "def_total_plays_faced"], errors="ignore"
    )

    return team_stats
