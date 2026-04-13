"""
================================================================================
MACRO ALPHA PROOF OF CONCEPT — USD/JPY Policy Divergence Analysis
================================================================================
Dual-axis visualization of USD/JPY vs. Fed-BoJ hawkish policy divergence,
combined with OLS regression for statistical validation of predictive power.

Mitigations Applied:
  - Stationarity:      Log returns instead of raw price levels
  - Look-Ahead Bias:   1-day lag on all independent variables
  - Multicollinearity:  Correlation check before interpreting coefficients
  - Scaling:           Secondary Y-axis for sentiment overlay
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "master_alpha_dataset.csv"
PLOT_OUTPUT = "macro_alpha_proof_of_concept.png"
MA_WINDOW = 20          # Smoothing window for divergence signal
LAG_DAYS = 1            # Predictive lag for regression


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load master dataset, compute log returns, apply lag, drop NaNs."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # --- Stationarity: compute 1-day log returns of USD/JPY ---
    df["usd_jpy_log_return"] = np.log(df["usd_jpy_close"] / df["usd_jpy_close"].shift(1))

    # --- Smoothed signal for visualization ---
    df["Divergence_MA20"] = df["Policy_Divergence_Hawk"].rolling(window=MA_WINDOW).mean()

    # --- Lag independent variables by 1 day to test PREDICTIVE power ---
    # This ensures we only use information available BEFORE the return period
    lagged_cols = ["Policy_Divergence_Hawk", "T10Y2Y", "vix"]
    for col in lagged_cols:
        df[f"{col}_lag{LAG_DAYS}"] = df[col].shift(LAG_DAYS)

    print(f"[OK] Dataset loaded: {df.shape[0]} rows")
    print(f"     Date range: {df.index.min().date()} → {df.index.max().date()}")

    return df


# =============================================================================
# TASK 1: DUAL-AXIS VISUALIZATION
# =============================================================================

def create_visualization(df: pd.DataFrame, output_path: str):
    """
    Dual-axis plot:
      Left Y-axis  → USD/JPY Close (price level, for narrative context)
      Right Y-axis → Policy_Divergence_Hawk 20-day MA (regime signal)
    Highlights the 2022-2024 divergence explosion.
    """
    # --- Style Setup ---
    plt.rcParams.update({
        "figure.facecolor": "#0a0e17",
        "axes.facecolor": "#0a0e17",
        "text.color": "#c8d6e5",
        "axes.labelcolor": "#c8d6e5",
        "xtick.color": "#636e80",
        "ytick.color": "#636e80",
        "axes.edgecolor": "#1e2a3a",
        "grid.color": "#1e2a3a",
        "grid.alpha": 0.5,
        "font.family": "monospace",
    })

    fig, ax1 = plt.subplots(figsize=(18, 8))

    # --- Highlight Zone: 2022-2024 Divergence Explosion ---
    highlight_start = datetime(2022, 1, 1)
    highlight_end = datetime(2024, 12, 31)
    ax1.axvspan(highlight_start, highlight_end, alpha=0.08, color="#e74c3c",
                label="2022–2024 Divergence Regime")

    # --- Left Axis: USD/JPY Price ---
    color_price = "#00d2ff"
    ax1.plot(df.index, df["usd_jpy_close"], color=color_price, linewidth=1.2,
             alpha=0.85, label="USD/JPY Close")
    ax1.set_xlabel("Date", fontsize=11, labelpad=10)
    ax1.set_ylabel("USD/JPY Close", color=color_price, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_price)
    ax1.set_ylim(df["usd_jpy_close"].min() * 0.95, df["usd_jpy_close"].max() * 1.05)
    ax1.grid(True, axis="both", linewidth=0.3)

    # --- Right Axis: Policy Divergence (Smoothed) ---
    ax2 = ax1.twinx()
    color_divergence = "#ff6b6b"
    ax2.plot(df.index, df["Divergence_MA20"], color=color_divergence, linewidth=1.8,
             alpha=0.9, label=f"Policy Divergence Hawk ({MA_WINDOW}d MA)")
    ax2.axhline(y=0, color="#636e80", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Policy Divergence (US_Hawk − BoJ_Hawk)", color=color_divergence,
                    fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_divergence)

    # --- Formatting ---
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Title ---
    fig.suptitle(
        "USD/JPY vs. Fed–BoJ Hawkish Policy Divergence",
        fontsize=16, fontweight="bold", color="#ffffff", y=0.97
    )
    ax1.set_title(
        "Does central bank narrative divergence predict currency moves?",
        fontsize=10, color="#636e80", pad=12
    )

    # --- Combined Legend ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc="upper left", framealpha=0.15, edgecolor="#1e2a3a",
               fontsize=9, facecolor="#0a0e17")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[OK] Visualization saved → {output_path}")


# =============================================================================
# TASK 2: STATISTICAL REGRESSION (OLS)
# =============================================================================

def run_regression(df: pd.DataFrame):
    """
    OLS Regression:
      Dependent variable:   USD/JPY 1-day log returns
      Independent variables: Lagged Policy Divergence, Lagged T10Y2Y, Lagged VIX
    
    All independent variables are lagged by 1 day to measure PREDICTIVE
    (not contemporaneous) power — this is the correct test for a trading signal.
    """
    # --- Prepare regression DataFrame, drop any NaN rows ---
    reg_cols = [
        "usd_jpy_log_return",
        f"Policy_Divergence_Hawk_lag{LAG_DAYS}",
        f"T10Y2Y_lag{LAG_DAYS}",
        f"vix_lag{LAG_DAYS}",
    ]
    reg_df = df[reg_cols].dropna()

    y = reg_df["usd_jpy_log_return"]
    X = reg_df[[
        f"Policy_Divergence_Hawk_lag{LAG_DAYS}",
        f"T10Y2Y_lag{LAG_DAYS}",
        f"vix_lag{LAG_DAYS}",
    ]]
    X = sm.add_constant(X)  # Add intercept

    print(f"\n[INFO] Regression sample: {len(reg_df)} observations")
    print(f"       Date range: {reg_df.index.min().date()} → {reg_df.index.max().date()}")

    # --- Multicollinearity Check ---
    print("\n" + "=" * 60)
    print("MULTICOLLINEARITY CHECK (Pairwise Correlations)")
    print("=" * 60)
    corr_cols = [
        f"Policy_Divergence_Hawk_lag{LAG_DAYS}",
        f"T10Y2Y_lag{LAG_DAYS}",
        f"vix_lag{LAG_DAYS}",
    ]
    corr_matrix = reg_df[corr_cols].corr()
    print(corr_matrix.round(4).to_string())

    # Flag high correlations
    for i, col_a in enumerate(corr_cols):
        for col_b in corr_cols[i + 1:]:
            r = corr_matrix.loc[col_a, col_b]
            if abs(r) > 0.8:
                print(f"\n[WARN] High correlation ({r:.4f}) between {col_a} and {col_b}")
                print("       Coefficient estimates may be unstable. Consider dropping one.")

    # --- OLS Regression ---
    print("\n" + "=" * 60)
    print("OLS REGRESSION RESULTS")
    print("=" * 60)
    model = sm.OLS(y, X).fit(cov_type="HC1")  # Heteroskedasticity-robust standard errors
    print(model.summary())

    # --- Clean Summary Table ---
    print("\n" + "=" * 60)
    print("COEFFICIENT SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-stat": model.tvalues,
        "P-value": model.pvalues,
        "Significant (5%)": ["Yes" if p < 0.05 else "No" for p in model.pvalues],
    })
    print(summary.round(6).to_string())
    print(f"\nR-squared:     {model.rsquared:.6f}")
    print(f"Adj R-squared: {model.rsquared_adj:.6f}")
    print(f"F-statistic:   {model.fvalue:.4f} (p = {model.f_pvalue:.6f})")
    print(f"Observations:  {int(model.nobs)}")

    return model


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def print_executive_summary(model):
    """Interpret results in plain language for the investment committee."""
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    div_col = f"Policy_Divergence_Hawk_lag{LAG_DAYS}"
    div_coef = model.params[div_col]
    div_pval = model.pvalues[div_col]
    div_sig = div_pval < 0.05

    print(f"""
QUESTION: Does the Fed-BoJ hawkish narrative divergence predict USD/JPY moves?

FINDING:
  The 1-day lagged Policy Divergence coefficient is {div_coef:.6f}
  with a p-value of {div_pval:.6f}.

INTERPRETATION:
  {"STATISTICALLY SIGNIFICANT at the 5% level." if div_sig else "NOT statistically significant at the 5% level."}
  
  {"A positive coefficient means that when the Fed is relatively more hawkish" if div_coef > 0 else "A negative coefficient means that when the BoJ is relatively more hawkish"}
  {"than the BoJ (higher divergence), USD/JPY tends to RISE the following day" if div_coef > 0 else "than the Fed (lower divergence), USD/JPY tends to FALL the following day"}
  {"— consistent with the rate-differential macro thesis." if div_sig else ""}

MODEL FIT:
  R-squared = {model.rsquared:.6f}
  This is {"low but EXPECTED for daily return prediction." if model.rsquared < 0.05 else "noteworthy for a daily frequency model."}
  Daily FX returns are dominated by noise; an R² above 0.01 at daily
  frequency with a significant alpha signal is a viable starting point
  for a systematic strategy when combined with proper risk management.

NEXT STEPS:
  1. Test with rolling/expanding window to check regime stability
  2. Add interaction terms (Divergence × VIX) for crisis regimes
  3. Backtest a simple long/short signal based on divergence thresholds
  4. Evaluate out-of-sample performance on 2025-2026 holdout data
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MACRO ALPHA PROOF OF CONCEPT")
    print("USD/JPY × Fed-BoJ Policy Divergence")
    print("=" * 60)

    # Load and prepare
    df = load_and_prepare(INPUT_CSV)

    # Task 1: Visualization
    create_visualization(df, PLOT_OUTPUT)

    # Task 2: OLS Regression
    model = run_regression(df)

    # Executive Summary
    print_executive_summary(model)

    print(f"\n[DONE] Analysis complete. Chart saved → {PLOT_OUTPUT}")