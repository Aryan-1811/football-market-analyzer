"""
data_loader.py
--------------
Wrapper around StatsBomb open data via statsbombpy.
Handles competition discovery, match selection, and event loading.

Dataset: https://github.com/statsbomb/open-data
Install:  pip install statsbombpy
"""

import pandas as pd
from statsbombpy import sb


# ── Competition helpers ───────────────────────────────────────────────

def get_free_competitions() -> pd.DataFrame:
    """Return all competitions available in StatsBomb open data."""
    comps = sb.competitions()
    return comps[["competition_id", "season_id", "competition_name",
                   "season_name", "competition_gender"]].sort_values(
        ["competition_name", "season_name"]
    ).reset_index(drop=True)


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a given competition / season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    cols = ["match_id", "match_date", "home_team", "away_team",
            "home_score", "away_score"]
    available = [c for c in cols if c in matches.columns]
    return matches[available].sort_values("match_date").reset_index(drop=True)


# ── Event loading ─────────────────────────────────────────────────────

def load_events(match_id: int) -> pd.DataFrame:
    """
    Load all events for a single match.
    Returns a cleaned DataFrame with key columns only.
    """
    events = sb.events(match_id=match_id)
    keep = [
        "id", "index", "period", "timestamp", "minute", "second",
        "type", "possession_team", "team", "player", "location",
        "shot_statsbomb_xg", "shot_outcome", "shot_body_part",
        "pass_outcome", "pass_length",
        "foul_committed_card", "bad_behaviour_card",
        "under_pressure", "duration",
    ]
    available = [c for c in keep if c in events.columns]
    return events[available].reset_index(drop=True)


def load_lineups(match_id: int) -> dict:
    """Return lineups dict keyed by team name."""
    return sb.lineups(match_id=match_id)


# ── Season aggregation ────────────────────────────────────────────────

def build_team_season_stats(competition_id: int, season_id: int,
                             max_matches: int = None) -> pd.DataFrame:
    """
    Aggregate team-level attack/defense stats across a season.
    Used to estimate lambda (expected goals) for the Poisson model.

    Returns DataFrame with avg_goals_for, avg_goals_against,
    avg_xg_for, avg_xg_against per team.
    """
    matches = get_matches(competition_id, season_id)
    if max_matches:
        matches = matches.head(max_matches)

    records = []
    print(f"Loading events for {len(matches)} matches...")

    for _, row in matches.iterrows():
        try:
            events = load_events(int(row["match_id"]))
        except Exception as e:
            print(f"  Skipping match {row['match_id']}: {e}")
            continue

        shots = events[events["type"] == "Shot"].copy()

        for team in [row["home_team"], row["away_team"]]:
            opponent = row["away_team"] if team == row["home_team"] else row["home_team"]
            team_shots = shots[shots["team"] == team]
            opp_shots  = shots[shots["team"] == opponent]

            records.append({
                "match_id":      row["match_id"],
                "team":          team,
                "opponent":      opponent,
                "is_home":       team == row["home_team"],
                "goals_for":     int((team_shots["shot_outcome"] == "Goal").sum()),
                "goals_against": int((opp_shots["shot_outcome"] == "Goal").sum()),
                "xg_for":   round(float(team_shots["shot_statsbomb_xg"].fillna(0).sum()), 3),
                "xg_against":round(float(opp_shots["shot_statsbomb_xg"].fillna(0).sum()), 3),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    agg = df.groupby("team").agg(
        matches_played=("match_id", "count"),
        goals_scored=("goals_for", "sum"),
        goals_conceded=("goals_against", "sum"),
        xg_for=("xg_for", "sum"),
        xg_against=("xg_against", "sum"),
    ).reset_index()

    agg["avg_goals_for"]     = (agg["goals_scored"]  / agg["matches_played"]).round(3)
    agg["avg_goals_against"] = (agg["goals_conceded"] / agg["matches_played"]).round(3)
    agg["avg_xg_for"]        = (agg["xg_for"]        / agg["matches_played"]).round(3)
    agg["avg_xg_against"]    = (agg["xg_against"]    / agg["matches_played"]).round(3)

    return agg.sort_values("avg_goals_for", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=== Free Competitions ===")
    comps = get_free_competitions()
    print(comps[["competition_name", "season_name"]].to_string(index=False))
    print("\n=== La Liga 2015/16 Matches (first 5) ===")
    matches = get_matches(competition_id=11, season_id=27)
    print(matches.head().to_string(index=False))
