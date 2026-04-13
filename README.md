# Macro Alpha Engine: Fed–BoJ Policy Divergence × USD/JPY

**End-to-End NLP Pipeline for Central Bank Sentiment Extraction, Time-Series Convergence & Alpha Validation**

---

## Abstract

This repository implements a full-cycle macro quantitative pipeline that extracts, scores, and backtests a currency alpha factor derived from central bank communication. 211 policy documents — 128 US Federal Reserve FOMC statements (2010–2026) and 83 Bank of Japan Summaries of Opinions (2016–2026) — are scraped, parsed, and scored for hawkish/dovish sentiment using a domain-specific keyword classifier. The resulting scores are merged with daily market data (USD/JPY, US 10Y-2Y spread, VIX, WTI Oil) via forward-fill regime propagation to construct the feature `Policy_Divergence_Hawk` $= S^{\text{hawk}}_{\text{US}} - S^{\text{hawk}}_{\text{BoJ}}$. Multi-horizon OLS regression reveals statistically significant predictive power at the 40-day forward horizon ($\beta = -0.0099$, $p = 0.0015$), consistent with a mean-reversion dynamic operating through institutional repositioning. A threshold-based backtest is evaluated with in-sample/out-of-sample splits and transaction cost adjustment.

## Data

| Parameter | Value |
|-----------|-------|
| Target | USD/JPY Close (daily) |
| Macro Controls | US 10Y-2Y Spread, VIX, WTI Oil |
| Text Sources | FOMC Statements (HTML), BoJ Summaries of Opinions (PDF) |
| Market Source | Yahoo Finance (`yfinance`), FRED (`fredapi`) |
| Text Storage | SQLite (`macro_engine.db`) |
| Period | 2010-01-01 → 2026-04-08 |
| Trading Days | 2,681 |
| Documents Scored | 211 (128 Fed + 83 BoJ) |

## Methodology

### 1. Data Engineering

Federal Reserve FOMC statements are scraped from `federalreserve.gov` using `requests` + `BeautifulSoup`, extracting text from the `#article` div. Bank of Japan Summaries of Opinions are downloaded as PDFs and parsed in-memory using `PyMuPDF` (fitz). All documents are stored in SQLite with `INSERT OR IGNORE` idempotency to prevent duplicates. Daily market data is pulled via `yfinance` (USD/JPY, VIX, Oil) and `fredapi` (10Y, 2Y Treasury yields).

### 2. Sentiment Scoring

Each document is scored for hawkish/dovish/neutral content using weighted keyword matching against curated phrase dictionaries for monetary policy language. For each document $d$:

$$S^{\text{hawk}}_d = \frac{\sum_{p \in \mathcal{H}} w_p \cdot c_p(d)}{\sum_{p \in \mathcal{H}} w_p \cdot c_p(d) + \sum_{p \in \mathcal{D}} w_p \cdot c_p(d)} \cdot \gamma(d)$$

Where $\mathcal{H}$ and $\mathcal{D}$ are the hawkish and dovish phrase sets, $w_p$ is the phrase weight (1–3), $c_p(d)$ is the count of phrase $p$ in document $d$, and $\gamma(d) = \min\left(\frac{\text{total\_weight}}{30}, 1\right)$ is a confidence scalar that prevents low-match documents from producing extreme scores.

An initial deployment of FinBERT-Tone (`yiyanghkust/finbert-tone`) with an overlapping chunking algorithm (400 words, 50-word overlap) achieved only 33% accuracy on known policy events — the model was trained on financial news, not central bank language. The keyword scorer achieves 100%.

### 3. Time-Series Convergence

The core alignment problem: market data updates 252 days/year, but central bank documents arrive ~8 times/year per institution. The NLP scores are joined onto the daily market calendar via `LEFT JOIN`, then forward-filled to propagate the most recent policy "regime" across all subsequent trading days until the next communication event. This models the step-function reality of monetary policy without introducing look-ahead bias.

The alpha feature is defined as:

$$\Delta^{\text{hawk}}_t = S^{\text{hawk}}_{\text{US},t} - S^{\text{hawk}}_{\text{BoJ},t}$$

### 4. Statistical Validation

OLS regression with heteroskedasticity-robust standard errors (HC1) is run on log returns:

$$r_t = \alpha + \beta_1 \Delta^{\text{hawk}}_{t-L} + \beta_2 \text{T10Y2Y}_{t-L} + \beta_3 \text{VIX}_{t-L} + \varepsilon_t$$

Where $L$ is the lag in trading days. All independent variables are lagged to ensure no look-ahead bias. Regressions are tested at daily, weekly, and monthly frequencies, across subsamples (2016–2021, 2022–2026), and with forward return horizons from 1 to 60 days.

### 5. Backtest

- **Signal**: Mean-reversion on policy divergence. When $\Delta^{\text{hawk}}_t$ exceeds its expanding median, short USD/JPY; when below, go long.
- **Rebalance**: Every 20 trading days.
- **Transaction costs**: 2bp per trade.
- **OOS split**: Train on 2016–2023, test on 2024–2026.

## Key Results

### Sentiment Validation

```
Events Tested:     9
Correct:           9
Wrong:             0
Accuracy:        100%
```

All 9 landmark policy events correctly classified, including the 2022 Fed tightening cycle (75bp hikes scored 0.67–0.77 hawkish), the September 2024 Fed pivot (scored 0.57 dovish), and the March 2024 BoJ rate hike (scored 0.88 hawkish).

### Multi-Lag Forward Returns

```
Horizon     Coefficient     t-stat      p-value     Sig
-------------------------------------------------------
1-day       +0.000058       +0.110      0.9122
5-day       −0.000820       −0.733      0.4635
10-day      −0.001736       −1.153      0.2490
20-day      −0.003689       −1.684      0.0922       *
40-day      −0.009923       −3.166      0.0015      ***
60-day      −0.009177       −2.385      0.0171       **
```

The negative coefficient at the 40-day horizon indicates a mean-reversion dynamic: periods of extreme Fed hawkishness relative to the BoJ ($\Delta^{\text{hawk}} \gg 0$) predict subsequent USD/JPY depreciation over 2–3 months, consistent with positioning unwinds after the market fully prices the rate differential.

### Backtest Performance

```
Metric                    Strategy        Buy & Hold
----------------------------------------------------
Total Return               40.02%           29.62%
Annual Return               3.22%            2.47%
Annualized Sharpe           −0.09            −0.17
Max Drawdown              −17.69%          −17.66%
Calmar Ratio                 0.18             0.14
Win Rate (daily)            52.3%            52.5%
Position Changes               28              N/A
OOS Sharpe (2024+)          +0.11            +0.11
```

![Backtest Results](backtest_results.png)

### Multi-Frequency Regression

```
Metric                          Daily       Weekly      Monthly
---------------------------------------------------------------
Observations                    2,567          538          124
R-squared                      0.0004       0.0009       0.0028
Divergence p-value             0.9122       0.8575       0.7733
```

No significance at any standard frequency — expected for a macro signal tested against daily/weekly noise. The signal's edge is in the multi-lag structure, not the frequency decomposition.

## Limitations

- **Keyword scorer ceiling**: While 100% accurate on known events, the keyword approach cannot capture surprise or deviation from market expectations — a statement that matches consensus perfectly carries no new information regardless of hawkishness.
- **BoJ document asymmetry**: 83 BoJ documents vs. 128 Fed creates an imbalance in regime estimation granularity on the BoJ side.
- **Forward-fill assumption**: The step-function regime model assumes policy posture is constant between meetings. In practice, speeches, minutes, and press conferences shift expectations intra-meeting.
- **OOS convergence**: The out-of-sample period (2024–2026) shows the strategy locked in a long position with no signal flips, producing returns identical to buy-and-hold. The short-side edge remains untested out-of-sample.
- **Negative Sharpe**: Full-sample Sharpe is negative against a 4% risk-free rate. The strategy underperforms T-bills as a standalone allocation.

## Potential Enhancements

| Approach | Rationale |
|----------|-----------|
| **Surprise Component** | Score divergence from OIS-implied expectations, not absolute hawkishness |
| **Change Signal** | First-difference of divergence (momentum) rather than level |
| **Event Study** | Restrict analysis to ±5 day windows around meeting dates |
| **Fine-Tuned LLM** | Train on labeled central bank text for nuanced tone detection |
| **Multi-Pair Extension** | Test signal on EUR/USD, GBP/USD, AUD/JPY for robustness |
| **Volatility Filter** | Reduce position size when VIX > 25 to avoid noise regimes |
| **Carry Overlay** | Combine with interest rate differential for multi-factor model |

## Project Structure

```
macro-alpha-engine/
├── market_data.py              # Yahoo Finance + FRED data pull
├── fed_scraper.py              # FOMC statement web crawler
├── boj_scraper.py              # BoJ PDF parser (PyMuPDF)
├── fill_fed_gap.py             # Fill Fed data gaps (2023-2026)
├── fill_boj_gap.py             # Fill BoJ data gaps (2018-2025)
├── sentiment_scoring.py        # FinBERT neural scorer
├── keyword_scorer.py           # Domain-specific keyword scorer
├── macro_convergence.py        # Merge + forward-fill + feature engineering
├── diagnostic_sentiment.py     # Validate scores against known events
├── macro_alpha_poc.py          # Dual-axis visualization + OLS regression
├── multi_freq_regression.py    # Daily/weekly/monthly frequency analysis
├── robustness_analysis.py      # Subsample, threshold, multi-lag tests
├── backtest.py                 # Strategy backtest with OOS split
├── macro_engine.db             # SQLite database (text + scores)
├── market_data_benchmark.csv   # Daily market data
└── master_alpha_dataset.csv    # Final merged dataset (12 features)
```

## Dependencies

```
python >= 3.10
yfinance
fredapi
requests
beautifulsoup4
PyMuPDF
torch
transformers
pandas
numpy
statsmodels
scipy
matplotlib
```

## Usage

```bash
git clone https://github.com/YOUR_USERNAME/macro-alpha-engine.git
cd macro-alpha-engine
pip install yfinance fredapi requests beautifulsoup4 PyMuPDF torch transformers pandas numpy statsmodels scipy matplotlib

# Phase 1: Data collection
python3 market_data.py
python3 fed_scraper.py && python3 fill_fed_gap.py
python3 boj_scraper.py && python3 fill_boj_gap.py

# Phase 2: Sentiment scoring
python3 keyword_scorer.py

# Phase 3: Convergence
python3 macro_convergence.py

# Phase 4: Validation & backtest
python3 diagnostic_sentiment.py
python3 macro_alpha_poc.py
python3 multi_freq_regression.py
python3 robustness_analysis.py
python3 backtest.py
```

## Disclaimer

This research is for educational and analytical purposes only. It does not constitute investment advice. Past performance on historical or simulated data does not guarantee future results. The authors assume no liability for financial losses incurred from the use of this code or methodology.

---

*Built as a macro quantitative research exercise.*
