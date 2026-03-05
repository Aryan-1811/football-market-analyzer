"""
inplay_engine.py
----------------
Simulates in-play odds movement using StatsBomb match event data.

Parses real match events (goals, red cards, half time) and
recalculates market odds after each key trigger -- mimicking
how a live trading desk adjusts prices during a match.

Key concepts:
  - Time-remaining factor: fewer minutes left = less variance remaining
  - Scoreline adjustment: leading teams sit back, trailing teams push forward
  - Red card penalty: reduces the affected team's expected goals rate
  - Over 2.5 / BTTS: correctly accounts for goals already scored
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.stats import poisson as sp
from src.poisson_model import compile_odds, MatchOdds


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Odds state captured at a single point in the match timeline."""
    minute:      int
    second:      int
    trigger:     str
    team:        Optional[str]
    home_score:  int
    away_score:  int
    home_reds:   int
    away_reds:   int
    odds:        MatchOdds
    xg_home:     float = 0.0
    xg_away:     float = 0.0


@dataclass
class InPlayTimeline:
    """Full record of odds snapshots across a match."""
    home_team:  str
    away_team:  str
    match_id:   int
    snapshots:  List[MarketSnapshot] = field(default_factory=list)

    def add(self, snapshot: MarketSnapshot):
        self.snapshots.append(snapshot)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert timeline to a flat DataFrame for analysis and plotting."""
        rows = []
        for s in self.snapshots:
            rows.append({
                "minute":      s.minute,
                "trigger":     s.trigger,
                "team":        s.team or "",
                "score":       f"{s.home_score}-{s.away_score}",
                "home_reds":   s.home_reds,
                "away_reds":   s.away_reds,
                "home_win":    s.odds.home_win_odds,
                "draw":        s.odds.draw_odds,
                "away_win":    s.odds.away_win_odds,
                "over25":      s.odds.over25_odds,
                "btts":        s.odds.btts_yes_odds,
                "home_win_p":  s.odds.home_win_prob,
                "draw_p":      s.odds.draw_prob,
                "away_win_p":  s.odds.away_win_prob,
                "xg_home":     s.xg_home,
                "xg_away":     s.xg_away,
            })
        return pd.DataFrame(rows)

    def print_report(self):
        """Print a formatted odds movement table to the terminal."""
        print(f"\n{'='*64}")
        print(f"  IN-PLAY TIMELINE: {self.home_team} vs {self.away_team}")
        print(f"{'='*64}")
        print(f"  {'MIN':<6} {'TRIGGER':<22} {'SCORE':<8} {'HW':>6} {'D':>6} {'AW':>6} {'O2.5':>6}")
        print(f"  {'-'*60}")
        for s in self.snapshots:
            team_label  = f"({s.team})" if s.team else ""
            trigger_str = f"{s.trigger} {team_label}".strip()
            print(
                f"  {s.minute:<6} {trigger_str:<22} "
                f"{s.home_score}-{s.away_score}      "
                f"{s.odds.home_win_odds:>6.2f} "
                f"{s.odds.draw_odds:>6.2f} "
                f"{s.odds.away_win_odds:>6.2f} "
                f"{s.odds.over25_odds:>6.2f}"
            )
        print(f"{'='*64}\n")


# ── Helper functions ──────────────────────────────────────────────────

def _desperation_factor(goals_ahead: int) -> float:
    """
    Adjust a team's attacking intent based on the scoreline.
    - Winning teams sit back slightly (factor < 1)
    - Losing teams push forward (factor > 1)
    """
    if goals_ahead > 0:
        return max(0.6, 1.0 - goals_ahead * 0.12)
    if goals_ahead < 0:
        return min(1.5, 1.0 + abs(goals_ahead) * 0.18)
    return 1.0


def _red_card_factor(red_cards: int) -> float:
    """Reduce a team's expected goals rate for each red card received."""
    return max(0.35, 1.0 - red_cards * 0.30)


# ── Main engine ───────────────────────────────────────────────────────

class InPlayEngine:
    """
    Processes a StatsBomb events DataFrame and generates a live odds
    timeline by recalculating prices after each key match event.

    Parameters
    ----------
    base_lambda_home : float  Pre-match expected goals for home team
    base_lambda_away : float  Pre-match expected goals for away team
    margin           : float  Bookmaker margin to apply (default 5%)
    rho              : float  Dixon-Coles correlation (default -0.1)

    Usage
    -----
        from src.data_loader import load_events
        from src.inplay_engine import InPlayEngine

        events = load_events(match_id=3788741)
        engine = InPlayEngine(base_lambda_home=1.8, base_lambda_away=1.2)
        timeline = engine.process(events, 3788741, "Barcelona", "Real Madrid")
        timeline.print_report()
        df = timeline.to_dataframe()
    """

    def __init__(self, base_lambda_home: float, base_lambda_away: float,
                 margin: float = 0.05, rho: float = -0.1):
        self.base_lh = base_lambda_home
        self.base_la = base_lambda_away
        self.margin  = margin
        self.rho     = rho

    def _recalculate(self, minute: int, home_score: int, away_score: int,
                     home_reds: int, away_reds: int) -> MatchOdds:
        """
        Recalculate live market odds given the current match state.

        Steps:
          1. Calculate remaining expected goals for each team
          2. Apply scoreline (desperation) and red card adjustments
          3. Run Poisson on remaining-match lambdas
          4. Shift 1X2 probabilities to reflect current scoreline
          5. Fix Over 2.5 and BTTS to account for goals already scored
          6. Apply bookmaker margin and return MatchOdds
        """
        mins_left          = max(1, 90 - minute)
        goal_diff          = home_score - away_score
        total_goals_so_far = home_score + away_score

        # ── Remaining expected goals ──────────────────────────────
        lh = (self.base_lh
              * _red_card_factor(home_reds)
              * _desperation_factor(goal_diff)
              * (mins_left / 90))

        la = (self.base_la
              * _red_card_factor(away_reds)
              * _desperation_factor(-goal_diff)
              * (mins_left / 90))

        lh, la = max(lh, 0.05), max(la, 0.05)

        # ── Compile odds for remaining match ──────────────────────
        remaining = compile_odds(lh, la, rho=self.rho)

        # ── 1X2 scoreline adjustment ──────────────────────────────
        hw_p = remaining.home_win_prob
        dr_p = remaining.draw_prob
        aw_p = remaining.away_win_prob

        if goal_diff > 0:
            # Home winning — draw in remaining time = home win overall
            hw_p = min(0.98, hw_p + dr_p * 0.6 + aw_p * 0.1)
            dr_p = max(0.01, dr_p * 0.3)
            aw_p = max(0.01, 1 - hw_p - dr_p)
        elif goal_diff < 0:
            # Away winning — draw in remaining time = away win overall
            aw_p = min(0.98, aw_p + dr_p * 0.6 + hw_p * 0.1)
            dr_p = max(0.01, dr_p * 0.3)
            hw_p = max(0.01, 1 - aw_p - dr_p)

        # Normalise to sum to 1
        total = hw_p + dr_p + aw_p
        hw_p /= total
        dr_p /= total
        aw_p /= total

        # ── Over 2.5 — account for goals already scored ──────────
        # Key fix: we only need (3 - goals_so_far) MORE goals
        goals_needed = max(0, 3 - total_goals_so_far)

        if goals_needed == 0:
            # 3 or more goals already scored — Over 2.5 is already won
            over25_p = 0.99

        elif goals_needed == 1:
            # Need just 1 more goal from either team in remaining time
            # P(at least 1 goal) = 1 - P(0 goals from both teams)
            prob_zero_home = sp.pmf(0, lh)
            prob_zero_away = sp.pmf(0, la)
            prob_no_goal   = prob_zero_home * prob_zero_away
            over25_p       = max(0.01, 1 - prob_no_goal)

        else:
            # Need 2+ more goals — use the remaining match over25 probability
            over25_p = max(0.01, remaining.over25_prob)

        # ── BTTS — account for goals already scored ───────────────
        # Key fix: if a team already scored, they don't need to score again
        home_already_scored = home_score > 0
        away_already_scored = away_score > 0

        if home_already_scored and away_already_scored:
            # Both already scored — BTTS Yes is already won
            btts_p = 0.99

        elif home_already_scored:
            # Home scored, just need away to score at least once
            # P(away scores >= 1) = 1 - P(away scores 0)
            btts_p = max(0.01, 1 - sp.pmf(0, la))

        elif away_already_scored:
            # Away scored, just need home to score at least once
            btts_p = max(0.01, 1 - sp.pmf(0, lh))

        else:
            # Neither scored yet — use remaining match btts probability
            btts_p = max(0.01, remaining.btts_prob)

        # ── Build final MatchOdds and apply margin ────────────────
        live = MatchOdds(
            home_win_prob = round(hw_p,     4),
            draw_prob     = round(dr_p,     4),
            away_win_prob = round(aw_p,     4),
            over25_prob   = round(over25_p, 4),
            btts_prob     = round(btts_p,   4),
            lambda_home   = round(lh, 3),
            lambda_away   = round(la, 3),
        )
        return live.with_margin(self.margin)

    def process(self, events: pd.DataFrame, match_id: int,
                home_team: str, away_team: str) -> InPlayTimeline:
        """
        Process a StatsBomb events DataFrame and return a full
        InPlayTimeline with odds snapshots at each key event.

        Triggers a recalculation on:
          - Kick Off (minute 0)
          - Every GOAL
          - Every RED CARD
          - Half Time
          - Every 15 minutes if no other trigger occurred
        """
        timeline = InPlayTimeline(
            home_team=home_team,
            away_team=away_team,
            match_id=match_id,
        )

        home_score = away_score = home_reds = away_reds = 0
        xg_home = xg_away = 0.0
        last_trigger_min = -1

        # Pre-match / kick-off snapshot
        prematch = self._recalculate(0, 0, 0, 0, 0)
        timeline.add(MarketSnapshot(
            minute=0, second=0, trigger="KO", team=None,
            home_score=0, away_score=0,
            home_reds=0,  away_reds=0,
            odds=prematch, xg_home=0.0, xg_away=0.0,
        ))

        for _, ev in events.sort_values("index").iterrows():
            etype  = ev.get("type", "")
            team   = ev.get("team", "")
            minute = int(ev.get("minute", 0))
            second = int(ev.get("second", 0))

            # Accumulate xG from shots
            xg = float(ev.get("shot_statsbomb_xg") or 0)
            if team == home_team:
                xg_home += xg
            else:
                xg_away += xg

            trigger = None

            # Goal
            if etype == "Shot" and ev.get("shot_outcome") == "Goal":
                if team == home_team:
                    home_score += 1
                else:
                    away_score += 1
                trigger = "GOAL"

            # Red card
            elif etype in ("Foul Committed", "Bad Behaviour"):
                col  = "foul_committed_card" if etype == "Foul Committed" else "bad_behaviour_card"
                card = str(ev.get(col) or "")
                if "Red" in card:
                    if team == home_team:
                        home_reds += 1
                    else:
                        away_reds += 1
                    trigger = "RED CARD"

            # Half time
            elif etype == "Half Time":
                trigger = "HT"

            # Periodic update every 15 minutes
            elif minute - last_trigger_min >= 15 and minute > 0:
                trigger = f"{minute}' UPDATE"

            if trigger:
                odds = self._recalculate(
                    minute, home_score, away_score, home_reds, away_reds
                )
                timeline.add(MarketSnapshot(
                    minute=minute, second=second,
                    trigger=trigger,
                    team=team if trigger in ("GOAL", "RED CARD") else None,
                    home_score=home_score, away_score=away_score,
                    home_reds=home_reds,   away_reds=away_reds,
                    odds=odds,
                    xg_home=round(xg_home, 3),
                    xg_away=round(xg_away, 3),
                ))
                last_trigger_min = minute

        return timeline