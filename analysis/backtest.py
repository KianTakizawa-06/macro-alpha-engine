"""
================================================================================
MACRO ALPHA BACKTEST — USD/JPY Policy Divergence Strategy
================================================================================
Backtests a simple long/short strategy based on the 40-day forward signal
discovered in the robustness analysis.

SIGNAL LOGIC:
  The regression found that Policy_Divergence_Hawk predicts USD/JPY returns
  over a 40-day horizon with a NEGATIVE coefficient (p=0.0015).
  
  Translation:
    - When Fed is more hawkish relative to BoJ (high divergence) →
      USD/JPY tends to FALL over next 40 days → SHORT USD/JPY
    - When BoJ is more hawkish relative to Fed (low divergence) →
      USD/JPY tends to RISE over next 40 days → LONG USD/JPY

STRATEGY:
  - Rebalance every 20 trading days (monthly, to avoid excessive turnover)
  - Position sizing: +1 (long) or -1 (short) based on divergence signal
  - Threshold: use median divergence as the signal crossover
  - No leverage, no partial sizing (pure directional signal test)
  - Transaction cost: 2bp per trade (conservative for spot FX)

BENCHMARKS:
  - Buy & Hold USD/JPY
  - Risk-free (flat)

Save as:  backtest.py
Run from: your macro_engine directory (needs master_alpha_dataset.csv)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "master_alpha_dataset.csv"
PLOT_OUTPUT = "backtest_results.png"

REBALANCE_FREQ = 20        # Trading days between rebalances
TRANSACTION_COST = 0.0002  # 2bp per trade (round trip = 4bp)
SIGNAL_COLUMN = "Policy_Divergence_Hawk"

# Out-of-sample split: train on 2016-2023, test on 2024-2026
OOS_START = "2024-01-01"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df["daily_return"] = df["usd_jpy_close"].pct_change()
    print(f"[OK] Loaded {len(df)} rows: {df.index.min().date()} → {df.index.max().date()}")
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on policy divergence.

    Signal logic (mean-reversion based on regression finding):
      - Divergence ABOVE threshold → market has overpriced USD strength
        → SHORT USD/JPY (position = -1)
      - Divergence BELOW threshold → market has overpriced JPY strength
        → LONG USD/JPY (position = +1)

    The threshold is computed using an EXPANDING window to avoid look-ahead
    bias — at each point we only use data available up to that date.
    """
    data = df.copy()

    # Expanding median: only uses data up to current date (no look-ahead)
    data["signal_threshold"] = data[SIGNAL_COLUMN].expanding(min_periods=60).median()

    # Raw signal: -1 if above threshold (short), +1 if below (long)
    data["raw_signal"] = np.where(
        data[SIGNAL_COLUMN] > data["signal_threshold"], -1, 1
    )

    # Only rebalance every N days
    data["position"] = np.nan
    rebalance_dates = data.index[::REBALANCE_FREQ]

    for date in rebalance_dates:
        if date in data.index:
            data.loc[date, "position"] = data.loc[date, "raw_signal"]

    # Forward-fill position between rebalance dates
    data["position"] = data["position"].ffill()

    # Drop rows before we have enough data for the expanding median
    data.dropna(subset=["position", "daily_return"], inplace=True)

    # Detect position changes for transaction costs
    data["position_change"] = data["position"].diff().abs()
    data["position_change"] = data["position_change"].fillna(0)

    print(f"[OK] Signals generated: {len(data)} trading days")
    print(f"     Position changes: {(data['position_change'] > 0).sum()}")
    print(f"     Long days: {(data['position'] == 1).sum()}  |  Short days: {(data['position'] == -1).sum()}")

    return data


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strategy returns, including transaction costs.
    Also compute buy-and-hold benchmark.
    """
    # Strategy return: position * daily_return - transaction costs on rebalance days
    data["strategy_gross"] = data["position"] * data["daily_return"]
    data["tc_cost"] = data["position_change"] * TRANSACTION_COST
    data["strategy_net"] = data["strategy_gross"] - data["tc_cost"]

    # Buy and hold benchmark
    data["buyhold_return"] = data["daily_return"]

    # Cumulative returns (equity curves)
    data["strategy_equity"] = (1 + data["strategy_net"]).cumprod()
    data["buyhold_equity"] = (1 + data["buyhold_return"]).cumprod()

    print(f"[OK] Backtest complete")
    return data


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_metrics(returns: pd.Series, label: str, risk_free_annual: float = 0.04) -> dict:
    """Compute standard quantitative performance metrics."""
    daily_rf = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = returns - daily_rf

    n_years = len(returns) / 252
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_annual) / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    # Sortino ratio (downside deviation only)
    downside = returns[returns < daily_rf] - daily_rf
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (annual_return - risk_free_annual) / downside_std if downside_std > 0 else 0

    return {
        "label": label,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_days": len(returns),
        "n_years": n_years,
    }


def print_metrics_table(metrics_list: list, section_label: str):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 75}")
    print(f"  {section_label}")
    print(f"{'=' * 75}")

    print(f"\n  {'Metric':<25s}", end="")
    for m in metrics_list:
        print(f"{m['label']:>20s}", end="")
    print()
    print("  " + "-" * 65)

    rows = [
        ("Total Return", "total_return", ".2%"),
        ("Annual Return", "annual_return", ".2%"),
        ("Annual Volatility", "annual_vol", ".2%"),
        ("Sharpe Ratio", "sharpe", ".3f"),
        ("Sortino Ratio", "sortino", ".3f"),
        ("Max Drawdown", "max_drawdown", ".2%"),
        ("Calmar Ratio", "calmar", ".3f"),
        ("Win Rate (daily)", "win_rate", ".1f"),
        ("Trading Days", "n_days", ".0f"),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<25s}", end="")
        for m in metrics_list:
            val = m[key]
            if fmt == ".1f" or fmt == ".0f":
                print(f"{val:>20{fmt}}", end="")
            else:
                print(f"{val:>20{fmt}}", end="")
        print()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_backtest_plot(data: pd.DataFrame, oos_start: str, output_path: str):
    """4-panel backtest visualization."""
    plt.rcParams.update({
        "figure.facecolor": "#0a0e17",
        "axes.facecolor": "#0a0e17",
        "text.color": "#c8d6e5",
        "axes.labelcolor": "#c8d6e5",
        "xtick.color": "#636e80",
        "ytick.color": "#636e80",
        "axes.edgecolor": "#1e2a3a",
        "grid.color": "#1e2a3a",
        "grid.alpha": 0.4,
        "font.family": "monospace",
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    oos_date = pd.Timestamp(oos_start)

    # --- Panel 1: Equity Curves ---
    ax1 = axes[0, 0]
    ax1.plot(data.index, data["strategy_equity"], color="#50fa7b", linewidth=1.5, label="Strategy (net)")
    ax1.plot(data.index, data["buyhold_equity"], color="#636e80", linewidth=1, alpha=0.7, label="Buy & Hold")
    ax1.axvline(oos_date, color="#ff6b6b", linewidth=1, linestyle="--", alpha=0.7, label="OOS Start")
    ax1.set_title("Equity Curves", fontsize=12, fontweight="bold", color="#ffffff")
    ax1.set_ylabel("Cumulative Return ($1 invested)")
    ax1.legend(fontsize=9, framealpha=0.2, loc="upper left")
    ax1.grid(True, linewidth=0.3)

    # --- Panel 2: Drawdown ---
    ax2 = axes[0, 1]
    strat_equity = data["strategy_equity"]
    strat_dd = (strat_equity - strat_equity.cummax()) / strat_equity.cummax()
    bh_equity = data["buyhold_equity"]
    bh_dd = (bh_equity - bh_equity.cummax()) / bh_equity.cummax()

    ax2.fill_between(data.index, strat_dd, 0, alpha=0.4, color="#50fa7b", label="Strategy")
    ax2.fill_between(data.index, bh_dd, 0, alpha=0.3, color="#636e80", label="Buy & Hold")
    ax2.axvline(oos_date, color="#ff6b6b", linewidth=1, linestyle="--", alpha=0.7)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold", color="#ffffff")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9, framealpha=0.2)
    ax2.grid(True, linewidth=0.3)

    # --- Panel 3: Rolling 60-day Sharpe ---
    ax3 = axes[1, 0]
    rolling_ret = data["strategy_net"].rolling(60).mean() * 252
    rolling_vol = data["strategy_net"].rolling(60).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol
    rolling_sharpe = rolling_sharpe.clip(-3, 3)  # cap for display

    ax3.plot(data.index, rolling_sharpe, color="#00d2ff", linewidth=0.8, alpha=0.8)
    ax3.axhline(0, color="#636e80", linewidth=0.5, linestyle="--")
    ax3.axhline(1, color="#50fa7b", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.axhline(-1, color="#ff6b6b", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.axvline(oos_date, color="#ff6b6b", linewidth=1, linestyle="--", alpha=0.7)
    ax3.set_title("Rolling 60-day Sharpe Ratio", fontsize=12, fontweight="bold", color="#ffffff")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.grid(True, linewidth=0.3)

    # --- Panel 4: Position and divergence ---
    ax4 = axes[1, 1]
    ax4.plot(data.index, data[SIGNAL_COLUMN], color="#ff6b6b", linewidth=0.6, alpha=0.6, label="Divergence")
    ax4.plot(data.index, data["signal_threshold"], color="#ffb86c", linewidth=1, alpha=0.8, label="Threshold (expanding median)")

    # Shade long/short periods
    long_mask = data["position"] == 1
    short_mask = data["position"] == -1
    ax4.fill_between(data.index, data[SIGNAL_COLUMN].min(), data[SIGNAL_COLUMN].max(),
                     where=long_mask, alpha=0.05, color="#50fa7b")
    ax4.fill_between(data.index, data[SIGNAL_COLUMN].min(), data[SIGNAL_COLUMN].max(),
                     where=short_mask, alpha=0.05, color="#ff6b6b")

    ax4.axvline(oos_date, color="#ff6b6b", linewidth=1, linestyle="--", alpha=0.7)
    ax4.set_title("Signal & Positions", fontsize=12, fontweight="bold", color="#ffffff")
    ax4.set_ylabel("Policy Divergence")
    ax4.legend(fontsize=8, framealpha=0.2)
    ax4.grid(True, linewidth=0.3)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "USD/JPY Policy Divergence Strategy — Backtest Results",
        fontsize=15, fontweight="bold", color="#ffffff", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[OK] Backtest plot saved → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("MACRO ALPHA BACKTEST")
    print("USD/JPY × Fed-BoJ Policy Divergence (40-day Forward Signal)")
    print("=" * 75)

    # Load and generate signals
    df = load_data()
    data = generate_signals(df)

    # Run backtest
    data = run_backtest(data)

    # Split in-sample / out-of-sample
    is_data = data[data.index < OOS_START]
    oos_data = data[data.index >= OOS_START]

    # Compute metrics
    print("\n" + "=" * 75)
    print("  PERFORMANCE ANALYSIS")
    print("=" * 75)

    # Full sample
    full_strat = compute_metrics(data["strategy_net"], "Strategy")
    full_bh = compute_metrics(data["buyhold_return"].dropna(), "Buy & Hold")
    print_metrics_table([full_strat, full_bh], "FULL SAMPLE")

    # In-sample
    if len(is_data) > 60:
        is_strat = compute_metrics(is_data["strategy_net"], "Strategy")
        is_bh = compute_metrics(is_data["buyhold_return"].dropna(), "Buy & Hold")
        print_metrics_table([is_strat, is_bh], f"IN-SAMPLE (before {OOS_START})")

    # Out-of-sample
    if len(oos_data) > 60:
        oos_strat = compute_metrics(oos_data["strategy_net"], "Strategy")
        oos_bh = compute_metrics(oos_data["buyhold_return"].dropna(), "Buy & Hold")
        print_metrics_table([oos_strat, oos_bh], f"OUT-OF-SAMPLE ({OOS_START} onward)")

    # Generate plot
    print("\n[5] Generating backtest visualization...")
    create_backtest_plot(data, OOS_START, PLOT_OUTPUT)

    # Final summary
    print("\n" + "=" * 75)
    print("  EXECUTIVE SUMMARY")
    print("=" * 75)

    oos_sharpe = oos_strat["sharpe"] if len(oos_data) > 60 else None

    print(f"""
  STRATEGY: Mean-reversion on Fed-BoJ hawkish policy divergence
  HORIZON:  Rebalance every {REBALANCE_FREQ} trading days
  COSTS:    {TRANSACTION_COST*10000:.0f}bp per trade

  FULL SAMPLE:
    Sharpe:   {full_strat['sharpe']:+.3f}
    Return:   {full_strat['annual_return']:+.2%} annualized
    Max DD:   {full_strat['max_drawdown']:.2%}
""")

    if oos_sharpe is not None:
        print(f"""  OUT-OF-SAMPLE ({OOS_START}+):
    Sharpe:   {oos_strat['sharpe']:+.3f}
    Return:   {oos_strat['annual_return']:+.2%} annualized
    Max DD:   {oos_strat['max_drawdown']:.2%}
""")

        if oos_strat["sharpe"] > 0.5:
            print("  VERDICT: The strategy shows positive out-of-sample performance.")
            print("           Consider further development with proper risk management.")
        elif oos_strat["sharpe"] > 0:
            print("  VERDICT: Positive but weak out-of-sample Sharpe.")
            print("           The signal has some value but needs enhancement — consider")
            print("           combining with carry, momentum, or volatility filters.")
        else:
            print("  VERDICT: Negative out-of-sample Sharpe.")
            print("           The in-sample signal did not generalize. The 40-day")
            print("           predictability may be a statistical artifact or the")
            print("           relationship may have shifted in the recent regime.")

    print(f"\n{'=' * 75}")
    print("[DONE] Backtest complete.")
    print(f"{'=' * 75}")