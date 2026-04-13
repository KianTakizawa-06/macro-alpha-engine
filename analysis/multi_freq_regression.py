"""
================================================================================
MULTI-FREQUENCY ALPHA ANALYSIS
================================================================================
Tests the Policy Divergence signal at daily, weekly, and monthly frequencies
to determine where the signal-to-noise ratio is strongest.

RATIONALE:
  Central banks meet ~8x per year. A policy shift announced on Day 1 doesn't
  move USD/JPY just on Day 2 — it reprices over weeks as the market digests
  the implications, analysts publish research, and positioning adjusts.

  Daily returns are dominated by microstructure noise (order flow, HFT,
  stop-hunting). At weekly and monthly frequency, the macro signal has
  time to compound and become visible above the noise floor.

Save as:  multi_freq_regression.py
Run from: your macro_engine directory (needs master_alpha_dataset.csv)
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "master_alpha_dataset.csv"
PLOT_OUTPUT = "multi_freq_analysis.png"
LAG_DAYS = 1


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    print(f"[OK] Loaded {len(df)} rows: {df.index.min().date()} → {df.index.max().date()}")
    return df


# =============================================================================
# RESAMPLING ENGINE
# =============================================================================

def build_frequency_datasets(df: pd.DataFrame) -> dict:
    """
    Build datasets at daily, weekly, and monthly frequency.

    For each frequency:
      - Resample price to period-end close
      - Resample sentiment to period-end value (last known regime)
      - Compute log returns over that period
      - Lag independent variables by 1 period
    """
    datasets = {}

    # --- DAILY (original) ---
    daily = df.copy()
    daily["log_return"] = np.log(daily["usd_jpy_close"] / daily["usd_jpy_close"].shift(1))

    # Lag all independent vars by 1 day
    for col in ["Policy_Divergence_Hawk", "Policy_Divergence_Dove", "T10Y2Y", "vix"]:
        if col in daily.columns:
            daily[f"{col}_lag"] = daily[col].shift(LAG_DAYS)

    daily.dropna(subset=["log_return", "Policy_Divergence_Hawk_lag"], inplace=True)
    datasets["Daily"] = daily

    # --- WEEKLY ---
    weekly = pd.DataFrame()
    weekly["usd_jpy_close"] = df["usd_jpy_close"].resample("W-FRI").last()

    # For sentiment: take the last value of the week (most recent regime)
    for col in ["Policy_Divergence_Hawk", "Policy_Divergence_Dove",
                "US_Hawkish", "BoJ_Hawkish", "US_Dovish", "BoJ_Dovish"]:
        if col in df.columns:
            weekly[col] = df[col].resample("W-FRI").last()

    # For macro: take the last value of the week
    for col in ["T10Y2Y", "vix", "oil_wti"]:
        if col in df.columns:
            weekly[col] = df[col].resample("W-FRI").last()

    weekly["log_return"] = np.log(weekly["usd_jpy_close"] / weekly["usd_jpy_close"].shift(1))

    for col in ["Policy_Divergence_Hawk", "Policy_Divergence_Dove", "T10Y2Y", "vix"]:
        if col in weekly.columns:
            weekly[f"{col}_lag"] = weekly[col].shift(1)  # 1 week lag

    weekly.dropna(subset=["log_return", "Policy_Divergence_Hawk_lag"], inplace=True)
    datasets["Weekly"] = weekly

    # --- MONTHLY ---
    monthly = pd.DataFrame()
    monthly["usd_jpy_close"] = df["usd_jpy_close"].resample("ME").last()

    for col in ["Policy_Divergence_Hawk", "Policy_Divergence_Dove",
                "US_Hawkish", "BoJ_Hawkish", "US_Dovish", "BoJ_Dovish"]:
        if col in df.columns:
            monthly[col] = df[col].resample("ME").last()

    for col in ["T10Y2Y", "vix", "oil_wti"]:
        if col in df.columns:
            monthly[col] = df[col].resample("ME").last()

    monthly["log_return"] = np.log(monthly["usd_jpy_close"] / monthly["usd_jpy_close"].shift(1))

    for col in ["Policy_Divergence_Hawk", "Policy_Divergence_Dove", "T10Y2Y", "vix"]:
        if col in monthly.columns:
            monthly[f"{col}_lag"] = monthly[col].shift(1)  # 1 month lag

    monthly.dropna(subset=["log_return", "Policy_Divergence_Hawk_lag"], inplace=True)
    datasets["Monthly"] = monthly

    for freq, data in datasets.items():
        print(f"  {freq:8s}: {len(data)} observations")

    return datasets


# =============================================================================
# REGRESSION ENGINE
# =============================================================================

def run_regression(data: pd.DataFrame, label: str) -> dict:
    """Run OLS with HC1 robust errors, return results dict."""
    iv_cols = []
    for col in ["Policy_Divergence_Hawk_lag", "T10Y2Y_lag", "vix_lag"]:
        if col in data.columns:
            iv_cols.append(col)

    reg_df = data[["log_return"] + iv_cols].dropna()

    y = reg_df["log_return"]
    X = sm.add_constant(reg_df[iv_cols])

    model = sm.OLS(y, X).fit(cov_type="HC1")

    # Extract divergence stats
    div_col = "Policy_Divergence_Hawk_lag"
    return {
        "label": label,
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_pvalue": model.f_pvalue,
        "div_coef": model.params.get(div_col, np.nan),
        "div_pvalue": model.pvalues.get(div_col, np.nan),
        "div_tstat": model.tvalues.get(div_col, np.nan),
        "t10y2y_coef": model.params.get("T10Y2Y_lag", np.nan),
        "t10y2y_pvalue": model.pvalues.get("T10Y2Y_lag", np.nan),
        "vix_coef": model.params.get("vix_lag", np.nan),
        "vix_pvalue": model.pvalues.get("vix_lag", np.nan),
        "model": model,
    }


def run_all_regressions(datasets: dict) -> list:
    """Run regressions at all frequencies and collect results."""
    results = []
    for freq, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  {freq.upper()} FREQUENCY REGRESSION")
        print(f"{'=' * 60}")
        res = run_regression(data, freq)
        results.append(res)

        # Print full summary for each
        print(res["model"].summary())

    return results


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(results: list):
    """Side-by-side comparison of all frequencies."""
    print("\n" + "=" * 80)
    print("MULTI-FREQUENCY COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<30s}", end="")
    for r in results:
        print(f"{r['label']:>16s}", end="")
    print()
    print("-" * 78)

    # Rows
    metrics = [
        ("Observations", "n_obs", "d"),
        ("R-squared", "r_squared", ".6f"),
        ("Adj R-squared", "adj_r_squared", ".6f"),
        ("F-test p-value", "f_pvalue", ".6f"),
        ("", None, ""),  # spacer
        ("Divergence Coef", "div_coef", ".6f"),
        ("Divergence t-stat", "div_tstat", ".3f"),
        ("Divergence p-value", "div_pvalue", ".6f"),
        ("", None, ""),
        ("T10Y2Y Coef", "t10y2y_coef", ".6f"),
        ("T10Y2Y p-value", "t10y2y_pvalue", ".6f"),
        ("", None, ""),
        ("VIX Coef", "vix_coef", ".6f"),
        ("VIX p-value", "vix_pvalue", ".6f"),
    ]

    for label, key, fmt in metrics:
        if key is None:
            print()
            continue
        print(f"{label:<30s}", end="")
        for r in results:
            val = r[key]
            if fmt == "d":
                print(f"{val:>16d}", end="")
            else:
                print(f"{val:>16{fmt}}", end="")
        print()

    # Significance flags
    print("\n" + "-" * 78)
    print(f"{'Divergence Significant (5%)':<30s}", end="")
    for r in results:
        sig = "YES ***" if r["div_pvalue"] < 0.05 else "no"
        print(f"{sig:>16s}", end="")
    print()

    print(f"{'Divergence Significant (10%)':<30s}", end="")
    for r in results:
        sig = "YES *" if r["div_pvalue"] < 0.10 else "no"
        print(f"{sig:>16s}", end="")
    print()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_multi_freq_plot(datasets: dict, results: list, output_path: str):
    """
    3-panel chart showing the relationship at each frequency:
      - Scatter plot of lagged divergence vs returns
      - Regression line overlay
      - R² and p-value annotations
    """
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {"Daily": "#00d2ff", "Weekly": "#ff6b6b", "Monthly": "#50fa7b"}

    for ax, (freq, data), res in zip(axes, datasets.items(), results):
        color = colors[freq]
        x = data["Policy_Divergence_Hawk_lag"]
        y = data["log_return"]

        # Scatter
        ax.scatter(x, y, alpha=0.25, s=10, color=color, edgecolors="none")

        # Regression line
        x_sorted = np.linspace(x.min(), x.max(), 100)
        X_line = sm.add_constant(x_sorted)

        # Simple bivariate for the visual line
        simple_model = sm.OLS(y, sm.add_constant(x)).fit()
        y_line = simple_model.predict(X_line)
        ax.plot(x_sorted, y_line, color="#ffffff", linewidth=2, alpha=0.8)

        # Zero lines
        ax.axhline(0, color="#636e80", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0, color="#636e80", linewidth=0.5, linestyle="--", alpha=0.5)

        # Labels
        ax.set_xlabel("Policy Divergence (Lagged)", fontsize=10)
        ax.set_ylabel(f"{freq} Log Return", fontsize=10)
        ax.set_title(f"{freq} Frequency", fontsize=13, fontweight="bold", color="#ffffff")
        ax.grid(True, linewidth=0.3)

        # Annotation box
        sig_label = "SIG" if res["div_pvalue"] < 0.05 else "n.s."
        textstr = (
            f"R² = {res['r_squared']:.4f}\n"
            f"Div coef = {res['div_coef']:.5f}\n"
            f"p = {res['div_pvalue']:.4f} [{sig_label}]\n"
            f"n = {res['n_obs']}"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="#1e2a3a", edgecolor="#636e80", alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=props, color="#c8d6e5")

    fig.suptitle(
        "Policy Divergence × USD/JPY Returns — Multi-Frequency Analysis",
        fontsize=15, fontweight="bold", color="#ffffff", y=1.02
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[OK] Multi-frequency plot saved → {output_path}")


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def print_executive_summary(results: list):
    """Plain-language interpretation for the investment committee."""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY — MULTI-FREQUENCY ANALYSIS")
    print("=" * 80)

    # Find best frequency
    best = min(results, key=lambda r: r["div_pvalue"])
    any_sig_05 = any(r["div_pvalue"] < 0.05 for r in results)
    any_sig_10 = any(r["div_pvalue"] < 0.10 for r in results)

    print(f"""
QUESTION: At what frequency does the Fed-BoJ policy divergence predict USD/JPY?

RESULTS BY FREQUENCY:""")

    for r in results:
        sig = ""
        if r["div_pvalue"] < 0.01:
            sig = "(*** significant at 1%)"
        elif r["div_pvalue"] < 0.05:
            sig = "(** significant at 5%)"
        elif r["div_pvalue"] < 0.10:
            sig = "(* significant at 10%)"
        else:
            sig = "(not significant)"

        direction = "positive" if r["div_coef"] > 0 else "negative"
        print(f"  {r['label']:8s}: coef = {r['div_coef']:+.6f}, p = {r['div_pvalue']:.4f} {sig}")

    print(f"""
STRONGEST SIGNAL: {best['label']} frequency
  Coefficient: {best['div_coef']:+.6f}
  P-value:     {best['div_pvalue']:.4f}
  R-squared:   {best['r_squared']:.4f}""")

    if any_sig_05:
        print("""
INTERPRETATION:
  The policy divergence signal IS statistically significant.
  This confirms the macro thesis: divergence in central bank hawkishness
  predicts USD/JPY directional moves at the frequency identified above.
  
  A positive coefficient means higher US hawkishness relative to BoJ
  predicts USD/JPY appreciation (dollar strengthening / yen weakening).
  
RECOMMENDED NEXT STEPS:
  1. Build a signal-based backtest at the significant frequency
  2. Test robustness with rolling 2-year windows
  3. Add interaction terms (Divergence × VIX regime)
  4. Evaluate transaction costs at the trading frequency""")

    elif any_sig_10:
        print("""
INTERPRETATION:
  The signal shows marginal significance at the 10% level.
  This is suggestive but not conclusive — the effect exists but is weak.
  
RECOMMENDED NEXT STEPS:
  1. Expand the BoJ document coverage (currently only 21 docs)
  2. Test with additional lags (2-week, 1-month forward returns)
  3. Consider a non-linear model (regime-switching)
  4. Add the dovish divergence as a second factor""")

    else:
        print("""
INTERPRETATION:
  No statistically significant relationship found at any frequency.
  
  HOWEVER — before concluding the thesis is dead, consider:
  
  1. DATA COVERAGE: Only 21 BoJ documents vs 128 Fed — the BoJ side
     of the divergence is severely undersampled. The signal may emerge
     with better BoJ coverage.
  
  2. NON-LINEARITY: The relationship may only activate in extreme
     divergence regimes (e.g., when divergence exceeds ±0.3).
     A linear OLS would miss this.
  
  3. REGIME DEPENDENCE: The signal may only work during tightening
     cycles (2022-2024) but not during stable periods (2016-2021).
     A full-sample regression dilutes this.
  
RECOMMENDED NEXT STEPS:
  1. PRIORITY: Fill BoJ data gaps — scrape all available BoJ docs
  2. Test subsample regression (2022-2026 only)
  3. Test threshold model: signal only when |divergence| > median
  4. Add multi-lag analysis: 1-day, 5-day, 20-day forward returns""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-FREQUENCY ALPHA ANALYSIS")
    print("USD/JPY × Fed-BoJ Policy Divergence")
    print("=" * 80)

    # Load
    df = load_data(INPUT_CSV)

    # Build frequency datasets
    print("\n[1] Building frequency datasets...")
    datasets = build_frequency_datasets(df)

    # Run regressions
    print("\n[2] Running regressions...")
    results = run_all_regressions(datasets)

    # Comparison table
    print_comparison(results)

    # Visualization
    print("\n[3] Generating multi-frequency scatter plots...")
    create_multi_freq_plot(datasets, results, PLOT_OUTPUT)

    # Executive summary
    print_executive_summary(results)

    print(f"\n{'=' * 80}")
    print("[DONE] Multi-frequency analysis complete.")
    print(f"{'=' * 80}")