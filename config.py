"""Configuration constants for the NFL Predictor pipeline."""

# Year range for historical data
YEARS = list(range(2015, 2026))  # 2015-2024

# Minimum year for walk-forward validation test sets
MIN_TEST_YEAR = 2018

# Play-by-play columns to load
PBP_COLUMNS = [
    # Game identifiers
    "game_id", "season", "week", "season_type", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    # Score
    "home_score", "away_score", "posteam_score", "defteam_score",
    # Play info
    "play_type", "down", "ydstogo", "yards_gained",
    "first_down", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    # Passing
    "pass_attempt", "complete_pass", "incomplete_pass", "interception",
    "pass_touchdown", "yards_after_catch", "air_yards",
    "sack", "qb_hit",
    "cpoe", "cp",
    # Rushing
    "rush_attempt", "rush_touchdown",
    # Turnovers
    "fumble", "fumble_lost",
    # Penalties
    "penalty", "penalty_yards",
    # EPA
    "epa", "qb_epa",
    "wp", "wpa",
    # Success
    "success",
    # Special teams
    "field_goal_attempt", "field_goal_result",
    "extra_point_attempt", "extra_point_result",
    "punt_attempt", "kickoff_attempt",
    # Situation
    "shotgun", "no_huddle", "qb_dropback", "qb_scramble",
    "play_type_nfl",
]

# Features used in the final matchup model
OFFENSIVE_FEATURES = [
    "off_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    "off_success_rate",
    "off_yards_per_play",
    "off_pass_yards_per_att",
    "off_rush_yards_per_att",
    "off_completion_pct",
    "off_cpoe",
    "off_third_down_conv_rate",
    "off_first_down_rate",
    "off_turnover_rate",
    "off_sack_rate_allowed",
    "off_scoring_rate",
    "off_pass_td_rate",
    "off_rush_td_rate",
    "off_penalty_rate",
    "off_no_huddle_rate",
]

DEFENSIVE_FEATURES = [
    "def_epa_per_play",
    "def_pass_epa_per_play",
    "def_rush_epa_per_play",
    "def_success_rate_allowed",
    "def_yards_per_play_allowed",
    "def_completion_pct_allowed",
    "def_third_down_conv_rate_allowed",
    "def_sack_rate",
    "def_turnover_forced_rate",
    "def_int_rate",
    "def_penalty_rate",
    "def_scoring_rate_allowed",
]

SPECIAL_TEAMS_FEATURES = [
    "st_fg_pct",
    "st_xp_pct",
    "st_punt_rate",
]

GAME_FEATURES = [
    "point_differential",
    "turnover_differential",
    "total_plays",
]

# All per-game team stat features
TEAM_STAT_FEATURES = (
    OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURES + GAME_FEATURES
)

# Advanced features added on top
ADVANCED_FEATURES = [
    "elo_rating",
    "elo_diff",
    "strength_of_schedule",
]

# Rolling window sizes
ROLLING_WINDOWS = [3, 5]

# Elo parameters
ELO_K = 20
ELO_HOME_ADVANTAGE = 48
ELO_MEAN = 1500
ELO_SEASON_REVERSION = 1 / 3

# Matchup contextual features
CONTEXT_FEATURES = [
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "is_playoff",
    "is_divisional",
    "spread_line",
]
