"""
poisson_model.py
----------------
Pre-match odds compiler using a Poisson goal model with
Dixon-Coles low-score correction.

Given team attack/defense ratings, computes:
  - Fair 1X2 probabilities and decimal odds
  - Over/Under 2.5 goals probabilities
  - BTTS (Both Teams To Score) probability
  - Score probability matrix
"""

import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass
from typing import Tuple
import pandas as pd


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class MatchOdds:
    home_win_prob:  float
    draw_prob:      float
    away_win_prob:  float
    over25_prob:    float
    btts_prob:      float
    lambda_home:    float
    lambda_away:    float

    def _safe_odds(self, prob: float) -> float:
        """Convert probability to decimal odds safely (avoids division by zero)."""
        if prob < 0.001:
            return 999.99
        return round(1 / prob, 2)

    # Fair decimal odds (no margin)
    @property
    def home_win_odds(self) -> float:
        return self._safe_odds(self.home_win_prob)

    @property
    def draw_odds(self) -> float:
        return self._safe_odds(self.draw_prob)

    @property
    def away_win_odds(self) -> float:
        return self._safe_odds(self.away_win_prob)

    @property
    def over25_odds(self) -> float:
        return self._safe_odds(self.over25_prob)

    @property
    def under25_odds(self) -> float:
        return self._safe_odds(1 - self.over25_prob)

    @property
    def btts_yes_odds(self) -> float:
        return self._safe_odds(self.btts_prob)

    @property
    def btts_no_odds(self) -> float:
        return self._safe_odds(1 - self.btts_prob)

    def with_margin(self, margin: float = 0.05) -> "MatchOdds":
        """Return a new MatchOdds with bookmaker margin (vig) applied."""
        scale = 1 + margin
        return MatchOdds(
            home_win_prob = min(self.home_win_prob * scale, 0.99),
            draw_prob     = min(self.draw_prob     * scale, 0.99),
            away_win_prob = min(self.away_win_prob * scale, 0.99),
            over25_prob   = min(self.over25_prob   * scale, 0.99),
            btts_prob     = min(self.btts_prob     * scale, 0.99),
            lambda_home   = self.lambda_home,
            lambda_away   = self.lambda_away,
        )

    def print_report(self, home: str = "Home", away: str = "Away"):
        print(f"\n{'='*50}")
        print(f"  {home} vs {away}")
        print(f"  λ Home: {self.lambda_home:.3f} | λ Away: {self.lambda_away:.3f}")
        print(f"{'='*50}")
        print(f"  1X2 Markets:")
        print(f"    Home Win  {self.home_win_prob*100:5.1f}%  →  {self.home_win_odds}")
        print(f"    Draw      {self.draw_prob*100:5.1f}%  →  {self.draw_odds}")
        print(f"    Away Win  {self.away_win_prob*100:5.1f}%  →  {self.away_win_odds}")
        print(f"\n  Totals / Specials:")
        print(f"    Over  2.5 {self.over25_prob*100:5.1f}%  →  {self.over25_odds}")
        print(f"    Under 2.5 {(1-self.over25_prob)*100:5.1f}%  →  {self.under25_odds}")
        print(f"    BTTS Yes  {self.btts_prob*100:5.1f}%  →  {self.btts_yes_odds}")
        print(f"    BTTS No   {(1-self.btts_prob)*100:5.1f}%  →  {self.btts_no_odds}")
        print(f"{'='*50}\n")


# ── Core Poisson functions ────────────────────────────────────────────

def _dixon_coles_tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    elif x == 1 and y == 0:
        return 1 + la * rho
    elif x == 0 and y == 1:
        return 1 + lh * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lambda_home: float, lambda_away: float,
                 max_goals: int = 8, rho: float = -0.1) -> np.ndarray:
    """
    Build a (max_goals+1) x (max_goals+1) matrix of score probabilities.
    Rows = home goals, Columns = away goals.
    Applies Dixon-Coles correction for scores <= 1-1.
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            p *= _dixon_coles_tau(h, a, lambda_home, lambda_away, rho)
            matrix[h, a] = p
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def compile_odds(lambda_home: float, lambda_away: float,
                 max_goals: int = 8, rho: float = -0.1) -> MatchOdds:
    """
    Compile a full set of pre-match market probabilities from
    team expected-goals parameters.

    Parameters
    ----------
    lambda_home : float  Expected goals for home team
    lambda_away : float  Expected goals for away team
    max_goals   : int    Upper truncation for score matrix
    rho         : float  Dixon-Coles correlation parameter

    Returns
    -------
    MatchOdds dataclass with probabilities and fair decimal odds
    """
    # Ensure lambdas never go too low (prevents near-zero probabilities)
    lambda_home = max(lambda_home, 0.05)
    lambda_away = max(lambda_away, 0.05)

    mat = score_matrix(lambda_home, lambda_away, max_goals, rho)

    home_win = float(np.tril(mat, -1).sum())
    draw     = float(np.trace(mat))
    away_win = float(np.triu(mat, 1).sum())

    over25 = sum(
        mat[h, a]
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
        if h + a > 2.5
    )

    btts = sum(
        mat[h, a]
        for h in range(1, max_goals + 1)
        for a in range(1, max_goals + 1)
    )

    # Ensure probabilities are never exactly zero
    over25 = max(float(over25), 0.001)
    btts   = max(float(btts),   0.001)

    return MatchOdds(
        home_win_prob = round(home_win, 4),
        draw_prob     = round(draw,     4),
        away_win_prob = round(away_win, 4),
        over25_prob   = round(over25,   4),
        btts_prob     = round(btts,     4),
        lambda_home   = round(lambda_home, 3),
        lambda_away   = round(lambda_away, 3),
    )


# ── Lambda estimation ─────────────────────────────────────────────────

def estimate_lambdas(home_team: str, away_team: str,
                     team_stats: pd.DataFrame,
                     home_advantage: float = 1.15) -> Tuple[float, float]:
    """
    Estimate expected goals (lambda) for each team using
    attack/defense ratings from season statistics.

    Formula (Dixon & Coles, 1997):
        lambda_home = home_att * away_def * league_avg * home_adv
        lambda_away = away_att * home_def * league_avg
    """
    stats = team_stats.set_index("team")
    league_avg = team_stats["avg_goals_for"].mean()

    def _rating(team, col, fallback):
        if team in stats.index:
            return stats.loc[team, col]
        print(f"  Warning: '{team}' not found in stats, using league average.")
        return fallback

    home_att = _rating(home_team, "avg_goals_for",     league_avg)
    home_def = _rating(home_team, "avg_goals_against", league_avg)
    away_att = _rating(away_team, "avg_goals_for",     league_avg)
    away_def = _rating(away_team, "avg_goals_against", league_avg)

    lh = (home_att / league_avg) * (away_def / league_avg) * league_avg * home_advantage
    la = (away_att / league_avg) * (home_def / league_avg) * league_avg

    # Floor at 0.5 to ensure realistic minimum expected goals
    return round(max(lh, 0.5), 3), round(max(la, 0.5), 3)


# ── Convenience wrapper ───────────────────────────────────────────────

def price_match(home_team: str, away_team: str,
                team_stats: pd.DataFrame,
                margin: float = 0.05,
                home_advantage: float = 1.15) -> MatchOdds:
    """
    One-shot convenience: estimate lambdas then compile all markets.
    Returns MatchOdds with bookmaker margin applied.
    """
    lh, la = estimate_lambdas(home_team, away_team, team_stats, home_advantage)
    fair = compile_odds(lh, la)
    return fair.with_margin(margin)


# ── CLI demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Demo: Barcelona vs Real Madrid")
    odds = compile_odds(lambda_home=1.80, lambda_away=1.35)
    odds.print_report("Barcelona", "Real Madrid")

    print("With 5% bookmaker margin:")
    odds.with_margin(0.05).print_report("Barcelona", "Real Madrid")