<![CDATA[<div align="center">

# 🏛️ Macro Alpha Engine

### Quantifying Central Bank Policy Divergence for USD/JPY Alpha Generation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Macro_Engine-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Time_Series-150458?style=flat-square&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Executive Summary

Central banks move currencies. When the Federal Reserve tightens while the Bank of Japan holds steady, the resulting policy divergence drives USD/JPY through interest rate differentials, carry trade flows, and institutional repositioning. But this divergence is communicated through dense, jargon-heavy policy documents published only 8 times per year — not through clean numerical data feeds. This project bridges that gap: it builds a fully automated pipeline that scrapes, parses, and scores every FOMC statement and BoJ Summary of Opinions since 2010 using NLP, then merges those sentiment signals with daily market data to construct a quantitative alpha factor.

The core engineered feature — `Policy_Divergence_Hawk`, defined as the difference between US and BoJ hawkish sentiment scores — was validated through OLS regression across multiple time frequencies and forward horizons. The signal shows statistically significant predictive power (p = 0.0015) for USD/JPY returns at the 40-day forward horizon, consistent with a macro factor that transmits through institutional repositioning rather than intraday noise. A full backtest of a mean-reversion strategy based on this signal is included, with in-sample/out-of-sample splits, transaction cost modeling, and standard risk-adjusted performance metrics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MACRO ALPHA ENGINE                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  PHASE 1: DATA ENGINEERING                                       │
  │                                                                  │
  │  Federal Reserve ──► requests + BeautifulSoup ──► HTML Parser    │
  │  (FOMC Statements)   Web Crawler (2010-2026)      128 Documents  │
  │                                                        │         │
  │  Bank of Japan ────► requests + PyMuPDF ──────► PDF Parser       │
  │  (Summary of          In-Memory Extraction       83 Documents    │
  │   Opinions)           (2016-2026)                      │         │
  │                                                        ▼         │
  │  Yahoo Finance ────► yfinance ─────────────────► market_data.csv │
  │  (USD/JPY, VIX,                                        │         │
  │   Oil WTI)                                             │         │
  │                                                        │         │
  │  FRED API ─────────► fredapi ──────────────────► (10Y, 2Y Yield) │
  │  (Treasury Yields)                                     │         │
  │                                                        ▼         │
  │                                              ┌─────────────────┐ │
  │                                              │ macro_engine.db │ │
  │                                              │    (SQLite)     │ │
  │                                              └────────┬────────┘ │
  └───────────────────────────────────────────────────────┼──────────┘
                                                          │
  ┌───────────────────────────────────────────────────────┼──────────┐
  │  PHASE 2: NLP SCORING ENGINE                          │          │
  │                                                       ▼          │
  │  ┌─────────────┐    ┌────────────────┐    ┌─────────────────┐   │
  │  │  Chunking   │───►│ FinBERT-Tone   │───►│  Hawkish Score  │   │
  │  │  Algorithm  │    │ (HuggingFace)  │    │  Dovish Score   │   │
  │  │ 400w / 50   │    │ BERT 512-token │    │  Neutral Score  │   │
  │  │   overlap   │    │  + Softmax     │    │  ──► SQLite     │   │
  │  └─────────────┘    └────────────────┘    └─────────────────┘   │
  │                                                                  │
  │  + Domain-Specific Keyword Scorer (BoJ/Fed policy vocabulary)    │
  └──────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────┼────────────────────────────────┐
  │  PHASE 3: ALPHA CONVERGENCE    │                                 │
  │                                ▼                                 │
  │  Daily Market Data ◄── LEFT JOIN + FORWARD FILL ──► NLP Scores  │
  │  (252 days/year)       (Regime Propagation)     (16 events/year) │
  │                                │                                 │
  │                                ▼                                 │
  │                  ┌──────────────────────────┐                    │
  │                  │   ALPHA FEATURES          │                    │
  │                  │                          │                    │
  │                  │   Policy_Divergence_Hawk │                    │
  │                  │   = US_Hawk - BoJ_Hawk   │                    │
  │                  │                          │                    │
  │                  │   Policy_Divergence_Dove │                    │
  │                  │   = US_Dove - BoJ_Dove   │                    │
  │                  └────────────┬─────────────┘                    │
  │                               │                                  │
  │                               ▼                                  │
  │  ┌────────────────────────────────────────────────────────┐      │
  │  │  VALIDATION & BACKTEST                                  │      │
  │  │  • OLS Regression (HC1 robust) on Log Returns          │      │
  │  │  • Multi-Frequency Analysis (Daily/Weekly/Monthly)      │      │
  │  │  • Multi-Lag Forward Returns (1d → 60d)                │      │
  │  │  • Subsample & Threshold Regime Tests                   │      │
  │  │  • Full Backtest w/ OOS Split & Transaction Costs       │      │
  │  └────────────────────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools | Purpose |
|-------|-------|---------|
| Data Extraction | `yfinance`, `fredapi`, `requests`, `BeautifulSoup` | Market data APIs, web scraping |
| Document Parsing | `PyMuPDF` (fitz) | In-memory PDF text extraction |
| Storage | `SQLite3` | Centralized text + score database |
| NLP / ML | `transformers`, `PyTorch`, FinBERT-Tone | Sentiment classification |
| Time Series | `pandas`, `numpy` | Merge, align, forward-fill, feature engineering |
| Statistics | `statsmodels`, `scipy` | OLS regression, hypothesis testing |
| Visualization | `matplotlib` | Dual-axis charts, backtest equity curves |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

### Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/macro-alpha-engine.git
cd macro-alpha-engine

# Install dependencies
pip install yfinance fredapi requests beautifulsoup4 PyMuPDF \
            torch transformers pandas numpy statsmodels scipy matplotlib
```

### Configuration

Set your FRED API key in the market data script:

```python
FRED_API_KEY = 'your_api_key_here'
```

---

## Quick Start

Run the pipeline in sequence. Each script is idempotent — safe to re-run.

```bash
# ── Phase 1: Data Engineering ──────────────────────────────
# Pull 16 years of daily market data (USD/JPY, yields, VIX, oil)
python3 market_data.py

# Scrape Federal Reserve FOMC statements (2010 → present)
python3 fed_scraper.py

# Scrape Bank of Japan Summary of Opinions PDFs (2016 → present)
python3 boj_scraper.py

# Fill any data gaps
python3 fill_fed_gap.py
python3 fill_boj_gap.py

# ── Phase 2: NLP Scoring ──────────────────────────────────
# Score all documents with keyword-based monetary policy scorer
python3 keyword_scorer.py

# (Optional) Score with FinBERT neural model
python3 sentiment_scoring.py

# ── Phase 3: Convergence & Analysis ───────────────────────
# Merge market data + NLP scores into master dataset
python3 macro_convergence.py

# Validate sentiment accuracy against known policy events
python3 diagnostic_sentiment.py

# ── Phase 4: Statistical Validation ───────────────────────
# Dual-axis visualization + OLS regression
python3 macro_alpha_poc.py

# Multi-frequency regression (daily/weekly/monthly)
python3 multi_freq_regression.py

# Robustness: subsample, threshold, multi-lag analysis
python3 robustness_analysis.py

# Full strategy backtest with OOS evaluation
python3 backtest.py
```

---

## Results

### Alpha Signal Visualization

USD/JPY price overlaid with the 20-day smoothed Policy Divergence signal, highlighting the 2022–2024 regime where Fed-BoJ hawkishness diverged most aggressively.

<!-- Replace with your generated chart -->
![Dual-Axis Visualization](macro_alpha_proof_of_concept.png)

### Sentiment Validation

The keyword-based scorer achieves **100% accuracy** on 9 landmark policy events, correctly identifying all major Fed rate hikes (2022 tightening cycle), the September 2024 Fed pivot to easing, the BoJ YCC band widening (Dec 2022), and the historic BoJ rate hikes in 2024.

| Event | Date | Expected | Score | Result |
|-------|------|----------|-------|--------|
| Fed: First rate hike (0→0.25%) | 2022-03-16 | Hawkish | 0.500 | ✅ |
| Fed: 75bp hike — largest since 1994 | 2022-06-15 | Hawkish | 0.767 | ✅ |
| Fed: Fourth consecutive 75bp hike | 2022-11-02 | Hawkish | 0.733 | ✅ |
| Fed: First rate cut (50bp) — pivot | 2024-09-18 | Dovish | 0.567 | ✅ |
| BoJ: YCC band widened | 2022-12-20 | Hawkish | 0.725 | ✅ |
| BoJ: First rate hike since 2007 | 2024-03-19 | Hawkish | 0.875 | ✅ |

### OLS Regression Findings

Linear regression on daily/weekly/monthly log returns showed no significance at short horizons — expected for a macro signal operating against daily microstructure noise. However, the **multi-lag forward return analysis** revealed statistically significant predictive power at longer horizons:

| Forward Horizon | Coefficient | t-stat | p-value | Significance |
|-----------------|-------------|--------|---------|--------------|
| 1-day | +0.000058 | +0.110 | 0.9122 | — |
| 5-day | −0.000820 | −0.733 | 0.4635 | — |
| 10-day | −0.001736 | −1.153 | 0.2490 | — |
| 20-day | −0.003689 | −1.684 | 0.0922 | * |
| **40-day** | **−0.009923** | **−3.166** | **0.0015** | **✱✱✱** |
| 60-day | −0.009177 | −2.385 | 0.0171 | ✱✱ |

The negative coefficient indicates a **mean-reversion dynamic**: periods of extreme Fed hawkishness relative to the BoJ predict subsequent USD/JPY weakness over the following 2–3 months, consistent with positioning unwinds after the market fully prices in the rate differential.

### Backtest Performance

<!-- Replace with your generated chart -->
![Backtest Results](backtest_results.png)

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Return | 40.02% | 29.62% |
| Annual Return | 3.22% | 2.47% |
| Sharpe Ratio | −0.087 | −0.169 |
| Max Drawdown | −17.69% | −17.66% |
| Calmar Ratio | 0.182 | 0.140 |
| Rebalance Frequency | 20 days | — |
| Transaction Cost | 2bp/trade | — |

---

## Project Structure

```
macro-alpha-engine/
│
├── market_data.py              # Phase 1: Yahoo Finance + FRED data pull
├── fed_scraper.py              # Phase 1: FOMC statement web crawler
├── boj_scraper.py              # Phase 1: BoJ PDF parser
├── fill_fed_gap.py             # Phase 1: Fill Fed data gaps (2023-2026)
├── fill_boj_gap.py             # Phase 1: Fill BoJ data gaps (2018-2025)
│
├── sentiment_scoring.py        # Phase 2: FinBERT neural scorer
├── keyword_scorer.py           # Phase 2: Domain-specific keyword scorer
│
├── macro_convergence.py        # Phase 3: Merge + forward-fill + feature engineering
├── diagnostic_sentiment.py     # Phase 3: Validate scores against known events
│
├── macro_alpha_poc.py          # Phase 4: Dual-axis viz + OLS regression
├── multi_freq_regression.py    # Phase 4: Daily/weekly/monthly frequency analysis
├── robustness_analysis.py      # Phase 4: Subsample, threshold, multi-lag tests
├── backtest.py                 # Phase 4: Full strategy backtest with OOS split
│
├── macro_engine.db             # SQLite database (text + scores)
├── market_data_benchmark.csv   # Daily market data
├── master_alpha_dataset.csv    # Final merged dataset (12 features)
│
└── README.md
```

---

## Key Design Decisions

**Why forward-fill?** Central banks meet ~8 times per year, but markets trade 252 days. Between meetings, the last known policy posture defines the monetary regime — a hawkish Fed statement doesn't expire the next day. Forward-filling models this step-function reality without introducing look-ahead bias.

**Why keywords over FinBERT?** FinBERT-tone was trained on financial news headlines ("Stock surges 10%"), not central bank language ("The Committee decided to raise the target range"). Our diagnostic showed FinBERT scored the June 2022 75bp hike — the largest in 28 years — as 0.07% hawkish. The domain-specific keyword scorer achieves 100% accuracy on landmark events.

**Why 40-day horizon?** Daily FX returns are dominated by microstructure noise. The multi-lag analysis revealed that the policy divergence signal needs ~2 months to fully transmit through analyst revisions, positioning adjustments, and carry trade flows — aligning naturally with the gap between central bank meetings.

---

## Disclaimer

This project is for **educational and research purposes only**. Nothing in this repository constitutes financial advice, investment recommendations, or a solicitation to buy or sell any financial instrument. Past performance does not guarantee future results. The statistical findings presented here are based on historical data and may not persist in live trading. Always consult a qualified financial advisor before making investment decisions. The authors assume no liability for any financial losses incurred from the use of this code or its outputs.

---

<div align="center">

Built with Python, PyTorch, and a deep appreciation for central bank communication patterns.

</div>
]]>