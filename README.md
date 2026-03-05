# ⚽ Football Betting Market Analyzer

A data-driven football betting market analysis tool built with **StatsBomb Open Data**.  
Simulates how a sportsbook trader prices, monitors, and adjusts markets based on real match event data.

---

## 🎯 What This Project Does

- Builds **pre-match odds** from a Poisson goal model using team attack/defense ratings
- Detects **in-play market triggers** (goals, red cards, half time) from StatsBomb event data
- Simulates **live odds movement** in response to match events — like a real trading desk
- Correctly reprices **Over 2.5** and **BTTS** markets accounting for goals already scored
- Identifies **sharp vs recreational betting patterns** using CLV (Closing Line Value)
- Produces a **full match trading report** with customer risk segmentation

---

## 📁 Project Structure
```
football-market-analyzer/
│
├── src/
│   ├── data_loader.py       # StatsBomb API wrapper — loads competitions, matches, events
│   ├── poisson_model.py     # Pre-match odds compiler (Poisson + Dixon-Coles correction)
│   ├── inplay_engine.py     # In-play odds movement engine
│   ├── market_monitor.py    # CLV analysis, bet pattern detection, customer segmentation
│   └── main.py              # Entry point — runs the full pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart
```bash
# Clone the repo
git clone https://github.com/yourusername/football-market-analyzer.git
cd football-market-analyzer

# Create and activate virtual environment
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main
```

---

## 📊 Data Source

Uses **StatsBomb Open Data** — free, no API key required.
- 📎 https://github.com/statsbomb/open-data
- Includes: La Liga 2015/16, Champions League, Euro 2024, Women's Super League, and more
- Event-level data: every pass, shot, tackle, foul and card in each match
```python
from statsbombpy import sb

# See all free competitions
comps = sb.competitions()

# Load match events
events = sb.events(match_id=69249)
```

---

## 🔬 Methodology

### 1. Pre-Match Odds Compiler
- Aggregates team attack and defense ratings from season event data
- Calculates expected goals (λ) per team using Dixon-Coles attack/defense ratings
- Applies Dixon-Coles low-score correction to the Poisson score matrix
- Converts score probability matrix into 1X2, Over/Under 2.5 and BTTS fair odds
- Applies bookmaker margin to produce final market prices

### 2. In-Play Market Movement Engine
- Parses StatsBomb event timeline chronologically
- Triggers odds recalculation on: **Goals, Red Cards, Half Time, every 15 minutes**
- Adjusts expected goals rate for: time remaining, scoreline desperation, red card penalties
- Over 2.5 and BTTS correctly account for goals already scored in the match
- Simulates live market suspension windows around key events

### 3. Betting Pattern & CLV Analysis
- Simulates bets placed at early, peak and closing timing windows
- Calculates **CLV (Closing Line Value)** — the gold standard measure of betting edge
- Calculates **EV (Expected Value)** per bet
- Produces full performance report: ROI, Sharpe ratio, max drawdown, CLV by timing
- Segments customers into Sharp, Value Hunter, Arber and Recreational categories

---

## 📈 Sample Output
```
==================================================
  Barcelona vs Real Madrid
  λ Home: 1.847 | λ Away: 1.203
==================================================
  1X2 Markets:
    Home Win   48.2%  →  2.08
    Draw       28.1%  →  3.56
    Away Win   23.7%  →  4.22

  Totals / Specials:
    Over  2.5  61.3%  →  1.63
    Under 2.5  38.7%  →  2.58
    BTTS Yes   58.4%  →  1.71
    BTTS No    41.6%  →  2.40
==================================================

IN-PLAY TIMELINE: Barcelona vs Real Madrid
================================================================
  MIN    TRIGGER              SCORE     HW      D     AW   O2.5
  0      KO                   0-0     2.08   3.56   4.22   1.63
  55     GOAL (Barcelona)     1-0     1.52   4.80   6.10   2.10
  61     GOAL (Real Madrid)   1-1     3.20   2.90   3.20   1.45
  76     GOAL (Barcelona)     2-1     1.35   5.80   9.20   1.02
================================================================

BETTING MARKET MONITOR — PERFORMANCE REPORT
========================================================
  Total Bets:        150
  ROI:               +8.3%
  Avg CLV:           +3.1%
  Sharpe Ratio:       1.42
  Max Drawdown:      -4.2 units
========================================================
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `statsbombpy` | StatsBomb open data loader |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `scipy` | Poisson distribution calculations |
| `Python 3.9+` | Core language |

---

## 💼 Relevance to Sports Trading

This project directly mirrors core responsibilities of a sportsbook trader:

| Trader Task | Project Component |
|---|---|
| Compiling pre-match prices | `poisson_model.py` |
| Managing in-play markets | `inplay_engine.py` |
| Monitoring betting patterns | `market_monitor.py` |
| Assessing customer risk | CLV and customer segmentation |
| Staying informed on match events | StatsBomb event parser |

---

## ⚙️ Configuration

All key parameters are set at the top of `main.py`:
```python
COMPETITION_ID = 11    # 11 = La Liga
SEASON_ID      = 27    # 27 = 2015/16 season
MATCH_INDEX    = 0     # which match to analyse (0 = first)
MARGIN         = 0.05  # bookmaker margin (5%)
MAX_MATCHES    = 20    # matches to use for season stats
N_BETS         = 150   # synthetic bets to simulate
```

To find all available competitions and their IDs:
```bash
python -c "from src.data_loader import get_free_competitions; print(get_free_competitions())"
```

---

## 📝 Licence

MIT — free to use and adapt.