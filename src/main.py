"""
main.py
-------
Entry point for the Football Betting Market Analyzer.

Runs the full pipeline:
  1. Load a StatsBomb match (La Liga by default)
  2. Build season team stats and estimate pre-match lambdas
  3. Compile pre-match odds (Poisson model)
  4. Process match events through the in-play engine
  5. Simulate bets and run market monitor analysis
  6. Print all reports to terminal

Usage:
    python src/main.py

To analyse a different match, change COMPETITION_ID, SEASON_ID, MATCH_INDEX.
Find available competitions by running: python src/data_loader.py
"""

from src.data_loader    import get_matches, load_events, build_team_season_stats
from src.poisson_model  import price_match
from src.inplay_engine  import InPlayEngine
from src.market_monitor import MarketMonitor

# ── Config ────────────────────────────────────────────────────────────
# La Liga 2015/16 (free in StatsBomb open data)
COMPETITION_ID = 11
SEASON_ID      = 27
MATCH_INDEX    = 0       # 0 = first match in the season
MARGIN         = 0.05    # 5% bookmaker margin
MAX_MATCHES    = 20      # how many matches to use for season stats
N_BETS         = 150     # synthetic bets to simulate


def main():
    print("\n" + "="*56)
    print("  FOOTBALL BETTING MARKET ANALYZER")
    print("  Data: StatsBomb Open Data | github.com/statsbomb/open-data")
    print("="*56)

    # ── Step 1: Load matches ──────────────────────────────────────────
    print("\n[1/5] Loading match list...")
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    match   = matches.iloc[MATCH_INDEX]
    home    = match["home_team"]
    away    = match["away_team"]
    mid     = int(match["match_id"])

    print(f"      Selected match: {home} vs {away} (ID: {mid})")
    print(f"      Date: {match.get('match_date', 'N/A')}")

    # ── Step 2: Build season stats ────────────────────────────────────
    print(f"\n[2/5] Building season team stats (first {MAX_MATCHES} matches)...")
    team_stats = build_team_season_stats(COMPETITION_ID, SEASON_ID,
                                          max_matches=MAX_MATCHES)
    print(f"      Built stats for {len(team_stats)} teams.")

    # ── Step 3: Pre-match odds ────────────────────────────────────────
    print(f"\n[3/5] Compiling pre-match odds...")
    prematch_odds = price_match(home, away, team_stats, margin=MARGIN)
    prematch_odds.print_report(home, away)

    # ── Step 4: In-play engine ────────────────────────────────────────
    print(f"[4/5] Processing match events (in-play engine)...")
    events = load_events(mid)
    print(f"      Loaded {len(events)} events.")

    engine   = InPlayEngine(
        base_lambda_home=prematch_odds.lambda_home,
        base_lambda_away=prematch_odds.lambda_away,
        margin=MARGIN,
    )
    timeline = engine.process(events, mid, home, away)
    timeline.print_report()

    # ── Step 5: Market monitor ────────────────────────────────────────
    print(f"[5/5] Running market monitor & betting pattern analysis...")
    tl_df   = timeline.to_dataframe()
    monitor = MarketMonitor(seed=42)
    bets    = monitor.simulate_bets(tl_df, f"{home} vs {away}", n_bets=N_BETS)

    print(monitor.generate_report(bets))

    print("\n  Customer Segmentation:")
    seg = monitor.customer_segmentation(bets)
    print(seg.to_string(index=False))

    print("\n[Done] Full pipeline complete.")
    print("       Explore further in: notebooks/market_analysis.ipynb\n")


if __name__ == "__main__":
    main()