"""Team abbreviation mappings and normalization."""

# Teams that relocated or rebranded during 2015-2024
TEAM_REMAP = {
    "OAK": "LV",   # Oakland Raiders → Las Vegas Raiders (2020)
    "SD": "LAC",    # San Diego Chargers → LA Chargers (2017)
    "STL": "LA",    # St. Louis Rams → LA Rams (2016)
}

# All 32 current NFL team abbreviations
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

# Division mappings for divisional game detection
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}

# Reverse mapping: team → division
TEAM_TO_DIVISION = {}
for div, teams in DIVISIONS.items():
    for team in teams:
        TEAM_TO_DIVISION[team] = div


def normalize_team(abbr: str) -> str:
    """Normalize a team abbreviation to current name."""
    return TEAM_REMAP.get(abbr, abbr)


def is_divisional_game(team1: str, team2: str) -> bool:
    """Check if two teams are in the same division."""
    t1 = normalize_team(team1)
    t2 = normalize_team(team2)
    return TEAM_TO_DIVISION.get(t1) == TEAM_TO_DIVISION.get(t2)
