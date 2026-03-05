"""
market_monitor.py
-----------------
Betting pattern analysis and Closing Line Value (CLV) detection.

CLV is the gold standard metric used by professional sports bettors
and sportsbook risk teams. If a bettor consistently beats the closing
line, they likely have a genuine edge -- and sportsbooks will limit them.

This module:
  1. Simulates bets placed at different timing windows (early/peak/closing)
  2. Calculates CLV and EV for each bet
  3. Generates a full performance report (ROI, Sharpe, drawdown)
  4. Segments hypothetical customers: Sharp / Value Hunter / Arber / Recreational

Key concept -- Closing Line Value (CLV):
  CLV = (closing_probability / taken_probability - 1) * 100
  Positive CLV = you got better odds than the market settled at = good signal.
  Negative CLV = the market moved against you = bad signal.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ── Bet data class ────────────────────────────────────────────────────

@dataclass
class SimulatedBet:
    """Represents a single simulated bet on a football market."""
    bet_id:         int
    match:          str
    market:         str           # home_win | draw | away_win | over25 | btts
    placement_time: str           # early | peak | closing
    odds_taken:     float         # odds at time of placement
    closing_odds:   float         # final odds before match starts
    stake:          float         # stake in units
    won:            bool
    clv:            float         # Closing Line Value %
    ev:             float         # Expected Value %

    @property
    def profit(self) -> float:
        """Profit in units (positive = win, negative = loss)."""
        return (self.odds_taken - 1) * self.stake if self.won else -self.stake

    @property
    def is_sharp(self) -> bool:
        """
        Flag as a sharp bet if:
          - CLV > 3% (genuinely beat the market)
          - Placed early (before the market has moved)
        Sportsbooks use signals like this to identify and limit sharp customers.
        """
        return self.clv > 3.0 and self.placement_time == "early"


# ── Core metric functions ─────────────────────────────────────────────

def calculate_clv(odds_taken: float, closing_odds: float) -> float:
    """
    Calculate Closing Line Value (CLV).

    Formula: CLV = (1/closing_odds) / (1/odds_taken) * 100 - 100

    Example:
      odds_taken   = 2.20  → implied prob = 45.5%
      closing_odds = 2.00  → implied prob = 50.0%
      CLV = 50.0 / 45.5 - 1 = +9.9%  (you got great value early)
    """
    taken_prob   = 1 / odds_taken
    closing_prob = 1 / closing_odds
    return round((closing_prob / taken_prob - 1) * 100, 3)


def calculate_ev(win_prob: float, odds: float) -> float:
    """
    Calculate Expected Value %.

    Formula: EV = (win_prob * (odds - 1) - (1 - win_prob)) * 100

    Positive EV = profitable bet in the long run.
    Negative EV = losing bet in the long run (how most recreational bets work).
    """
    return round((win_prob * (odds - 1) - (1 - win_prob)) * 100, 3)


# ── Main monitor class ────────────────────────────────────────────────

class MarketMonitor:
    """
    Simulates a betting market around StatsBomb-derived odds data.

    Generates synthetic bets at early/peak/closing windows, then
    analyses the patterns to produce a full trading report and
    customer risk segmentation.

    Usage
    -----
        from src.market_monitor import MarketMonitor

        # timeline_df is from InPlayTimeline.to_dataframe()
        monitor = MarketMonitor(seed=42)
        bets = monitor.simulate_bets(timeline_df, "Barcelona vs Real Madrid", n_bets=150)
        print(monitor.generate_report(bets))
        seg = monitor.customer_segmentation(bets)
        print(seg)
    """

    MARKETS   = ["home_win", "draw", "away_win", "over25", "btts"]
    BET_TIMES = ["early", "peak", "closing"]

    # Mapping from market name to probability column in timeline DataFrame
    PROB_COLS = {
        "home_win": "home_win_p",
        "draw":     "draw_p",
        "away_win": "away_win_p",
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _simulate_single_bet(self, bet_id: int, match: str, market: str,
                              tl: pd.DataFrame) -> Optional[SimulatedBet]:
        """
        Simulate one bet on a given market using the odds timeline.

        Timing windows:
          early   → odds at match kick-off (opening line)
          peak    → odds at mid-point of timeline
          closing → odds at final snapshot
        """
        if tl.empty or market not in tl.columns:
            return None

        odds_open  = float(tl.iloc[0][market])
        odds_close = float(tl.iloc[-1][market])

        # Choose a timing with realistic probabilities
        timing = self.rng.choice(self.BET_TIMES, p=[0.35, 0.35, 0.30])

        if timing == "early":
            # Early bets have slight variance around opening price
            odds_taken = odds_open * (1 + self.rng.normal(0, 0.02))
        elif timing == "peak":
            mid_row    = tl.iloc[len(tl) // 2]
            odds_taken = float(mid_row[market]) * (1 + self.rng.normal(0, 0.015))
        else:
            odds_taken = odds_close * (1 + self.rng.normal(0, 0.01))

        odds_taken = max(1.02, round(odds_taken, 2))

        # Win probability from model (use closing prob column if available)
        prob_col = self.PROB_COLS.get(market)
        if prob_col and prob_col in tl.columns:
            win_prob = float(tl.iloc[-1][prob_col])
        else:
            # Fallback: implied from closing odds with 5% margin removed
            win_prob = (1 / odds_close) * 0.95

        won   = bool(self.rng.random() < win_prob)
        stake = round(float(self.rng.uniform(5, 100)), 2)
        clv   = calculate_clv(odds_taken, odds_close)
        ev    = calculate_ev(win_prob, odds_taken)

        return SimulatedBet(
            bet_id=bet_id, match=match, market=market,
            placement_time=timing, odds_taken=odds_taken,
            closing_odds=odds_close, stake=stake,
            won=won, clv=clv, ev=ev,
        )

    def simulate_bets(self, timeline_df: pd.DataFrame,
                      match_label: str = "Match",
                      n_bets: int = 150) -> List[SimulatedBet]:
        """
        Simulate n_bets across all available markets on a match timeline.

        Parameters
        ----------
        timeline_df  : pd.DataFrame   Output of InPlayTimeline.to_dataframe()
        match_label  : str            Human-readable match name
        n_bets       : int            Number of synthetic bets

        Returns
        -------
        List[SimulatedBet]
        """
        bets = []
        for i in range(n_bets):
            market = str(self.rng.choice(self.MARKETS))
            bet = self._simulate_single_bet(i, match_label, market, timeline_df)
            if bet:
                bets.append(bet)
        return bets

    def bets_to_dataframe(self, bets: List[SimulatedBet]) -> pd.DataFrame:
        """Convert a list of SimulatedBet objects into a flat DataFrame."""
        return pd.DataFrame([{
            "bet_id":         b.bet_id,
            "match":          b.match,
            "market":         b.market,
            "placement_time": b.placement_time,
            "odds_taken":     b.odds_taken,
            "closing_odds":   b.closing_odds,
            "stake":          b.stake,
            "won":            b.won,
            "profit":         round(b.profit, 2),
            "clv":            b.clv,
            "ev":             b.ev,
            "is_sharp":       b.is_sharp,
        } for b in bets])

    def generate_report(self, bets: List[SimulatedBet]) -> str:
        """
        Generate a full text performance report.

        Includes: ROI, win rate, CLV, Sharpe ratio, max drawdown,
        CLV by timing window, ROI by market, and interpretation.
        """
        if not bets:
            return "No bets to analyse."

        df = self.bets_to_dataframe(bets)

        # ── Top-level metrics ──────────────────────────────────────
        total_bets   = len(df)
        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi          = (total_profit / total_staked) * 100
        win_rate     = df["won"].mean() * 100
        avg_clv      = df["clv"].mean()
        avg_ev       = df["ev"].mean()
        sharp_count  = df["is_sharp"].sum()
        sharp_pct    = (sharp_count / total_bets) * 100

        # ── Sharpe ratio ───────────────────────────────────────────
        # Using bet-level returns (profit / stake) as daily approximation
        returns = df["profit"] / df["stake"]
        sharpe  = (returns.mean() / returns.std() * (252 ** 0.5)
                   if returns.std() > 0 else 0.0)

        # ── Max drawdown ───────────────────────────────────────────
        cumulative  = df["profit"].cumsum()
        rolling_max = cumulative.cummax()
        max_dd      = (cumulative - rolling_max).min()

        # ── CLV by timing ──────────────────────────────────────────
        clv_by_timing = df.groupby("placement_time")["clv"].mean().round(2)

        # ── ROI by market ──────────────────────────────────────────
        roi_by_market = (
            df.groupby("market").apply(
                lambda g: g["profit"].sum() / g["stake"].sum() * 100
            ).round(1)
        )

        lines = [
            "",
            "=" * 56,
            "  BETTING MARKET MONITOR — PERFORMANCE REPORT",
            "=" * 56,
            f"  Total Bets:        {total_bets}",
            f"  Total Staked:      {total_staked:.1f} units",
            f"  Total P&L:         {total_profit:+.2f} units",
            f"  ROI:               {roi:+.1f}%",
            f"  Win Rate:          {win_rate:.1f}%",
            f"  Avg CLV:           {avg_clv:+.2f}%",
            f"  Avg EV:            {avg_ev:+.2f}%",
            f"  Sharpe Ratio:      {sharpe:.2f}",
            f"  Max Drawdown:      {max_dd:.2f} units",
            f"  Sharp Bets:        {sharp_count} ({sharp_pct:.1f}%)",
            "",
            "  CLV by Bet Timing:",
        ]

        for timing, clv_val in clv_by_timing.items():
            tag = "  <- sharp signal" if timing == "early" and clv_val > 3 else ""
            lines.append(f"    {timing:<12}  {clv_val:+.2f}%{tag}")

        lines += ["", "  ROI by Market:"]
        for market, roi_val in roi_by_market.items():
            lines.append(f"    {market:<15}  {roi_val:+.1f}%")

        lines += ["", "  Interpretation:"]
        if avg_clv > 3:
            lines.append("  [+] Positive avg CLV -- model has genuine pre-match edge.")
        elif avg_clv > 0:
            lines.append("  [~] Marginal CLV -- some edge but thin. Improve model.")
        else:
            lines.append("  [-] Negative CLV -- model underperforms vs the market.")

        if sharp_pct > 25:
            lines.append(f"  [!] High sharp ratio ({sharp_pct:.0f}%) -- risk flag for sportsbook.")

        lines.append("=" * 56)
        return "\n".join(lines)

    def customer_segmentation(self, bets: List[SimulatedBet]) -> pd.DataFrame:
        """
        Segment hypothetical customers by their betting behaviour.

        This mirrors what a sportsbook risk/trading team does to
        assess whether a customer is sharp (to be limited) or
        recreational (to be retained and marketed to).

        Segments
        --------
        Sharp        : High CLV, bets early, consistent profit
        Value Hunter : Positive CLV but small stakes
        Arber        : Very high win rate and CLV (likely arbitraging)
        Recreational : Low/negative CLV, bets at peak or closing

        Returns a DataFrame of customer profiles sorted by avg_clv.
        """
        df = self.bets_to_dataframe(bets)

        # Assign synthetic customer IDs (in real use, these come from a DB)
        df["customer_id"] = self.rng.integers(1, 20, size=len(df))

        cust = df.groupby("customer_id").agg(
            total_bets   =("bet_id",         "count"),
            avg_stake    =("stake",           "mean"),
            total_profit =("profit",          "sum"),
            avg_clv      =("clv",             "mean"),
            win_rate     =("won",             "mean"),
            early_pct    =("placement_time",  lambda x: (x == "early").mean()),
            sharp_bets   =("is_sharp",        "sum"),
        ).reset_index()

        def _segment(row) -> str:
            if row["avg_clv"] > 4 and row["early_pct"] > 0.5:
                return "Sharp"
            if row["avg_clv"] > 3 and row["win_rate"] > 0.58:
                return "Arber"
            if row["avg_clv"] > 2 and row["avg_stake"] < 15:
                return "Value Hunter"
            return "Recreational"

        cust["segment"] = cust.apply(_segment, axis=1)
        cust["roi"] = (
            cust["total_profit"] / (cust["total_bets"] * cust["avg_stake"]) * 100
        ).round(1)

        cols = ["customer_id", "segment", "total_bets", "avg_stake",
                "avg_clv", "roi", "win_rate", "sharp_bets"]
        return cust[cols].sort_values("avg_clv", ascending=False).reset_index(drop=True)


# ── CLI demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic timeline -- normally from InPlayTimeline.to_dataframe()
    synthetic_tl = pd.DataFrame([
        {"minute": 0,  "home_win": 2.05, "draw": 3.40, "away_win": 3.80,
         "over25": 1.72, "btts": 1.68,
         "home_win_p": 0.488, "draw_p": 0.294, "away_win_p": 0.263},
        {"minute": 23, "home_win": 1.55, "draw": 4.20, "away_win": 6.50,
         "over25": 1.65, "btts": 1.60,
         "home_win_p": 0.645, "draw_p": 0.238, "away_win_p": 0.154},
        {"minute": 67, "home_win": 1.22, "draw": 6.80, "away_win": 12.00,
         "over25": 1.55, "btts": 1.52,
         "home_win_p": 0.820, "draw_p": 0.147, "away_win_p": 0.083},
        {"minute": 90, "home_win": 1.08, "draw": 9.50, "away_win": 18.00,
         "over25": 1.45, "btts": 1.44,
         "home_win_p": 0.926, "draw_p": 0.105, "away_win_p": 0.056},
    ])

    monitor = MarketMonitor(seed=42)
    bets = monitor.simulate_bets(synthetic_tl, "Barcelona vs Real Madrid", n_bets=120)

    print(monitor.generate_report(bets))

    print("\n  Customer Segmentation:")
    seg = monitor.customer_segmentation(bets)
    print(seg.to_string(index=False))