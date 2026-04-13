"""
================================================================================
ROBUSTNESS & REGIME ANALYSIS
================================================================================
Three final tests to determine if the policy divergence signal has any
exploitable relationship with USD/JPY before closing the investigation.

TEST 1: SUBSAMPLE REGRESSION (2022-2026 only)
  The active policy divergence period. If the signal only works during
  aggressive tightening cycles, the full-sample OLS would miss it.

TEST 2: THRESHOLD / NON-LINEAR MODEL
  Tests whether extreme divergence moves (top/bottom quartile) predict
  returns, even if the average relationship is flat.

TEST 3: MULTI-LAG FORWARD RETURNS
  Tests 1-day, 5-day, 10-day, and 20-day forward returns.
  A macro signal may need weeks to fully transmit into FX prices.

Save as:  robustness_analysis.py
Run from: your macro_engine directory (needs master_alpha_dataset.csv)
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

INPUT_CSV = "master_alpha_dataset.csv"
PLOT_OUTPUT = "robustness_analysis.png"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    print(f"[OK] Loaded {len(df)} rows: {df.index.min().date()} → {df.index.max().date()}")
    return df


# =============================================================================
# TEST 1: SUBSAMPLE REGRESSION
# =============================================================================

def test_subsample(df: pd.DataFrame):
    """Run OLS on the active regime period only (2022-2026)."""
    print("\n" + "=" * 70)
    print("TEST 1: SUBSAMPLE REGRESSION (Active Regime Periods)")
    print("=" * 70)

    subsamples = {
        "Full Sample (2016-2026)": df,
        "Pre-Divergence (2016-2021)": df[df.index < "2022-01-01"],
        "Active Regime (2022-2026)": df[df.index >= "2022-01-01"],
        "Peak Divergence (2022-2024)": df[(df.index >= "2022-01-01") & (df.index < "2025-01-01")],
    }

    results = []

    for label, subset in subsamples.items():
        sub = subset.copy()
        sub["log_return"] = np.log(sub["usd_jpy_close"] / sub["usd_jpy_close"].shift(1))
        sub["div_lag"] = sub["Policy_Divergence_Hawk"].shift(1)
        sub["t10y2y_lag"] = sub["T10Y2Y"].shift(1)
        sub["vix_lag"] = sub["vix"].shift(1)

        reg = sub[["log_return", "div_lag", "t10y2y_lag", "vix_lag"]].dropna()

        if len(reg) < 20:
            print(f"\n  {label}: Too few observations ({len(reg)}), skipping")
            continue

        y = reg["log_return"]
        X = sm.add_constant(reg[["div_lag", "t10y2y_lag", "vix_lag"]])
        model = sm.OLS(y, X).fit(cov_type="HC1")

        r = {
            "label": label,
            "n": int(model.nobs),
            "r2": model.rsquared,
            "div_coef": model.params.get("div_lag", np.nan),
            "div_pval": model.pvalues.get("div_lag", np.nan),
            "div_tstat": model.tvalues.get("div_lag", np.nan),
        }
        results.append(r)

        sig = "***" if r["div_pval"] < 0.01 else "**" if r["div_pval"] < 0.05 else "*" if r["div_pval"] < 0.10 else ""
        print(f"\n  {label}")
        print(f"    n={r['n']:>5d}  R²={r['r2']:.4f}  "
              f"Div_coef={r['div_coef']:+.6f}  t={r['div_tstat']:+.3f}  p={r['div_pval']:.4f} {sig}")

    return results


# =============================================================================
# TEST 2: THRESHOLD / NON-LINEAR MODEL
# =============================================================================

def test_threshold(df: pd.DataFrame):
    """Test if extreme divergence moves have stronger predictive power."""
    print("\n" + "=" * 70)
    print("TEST 2: THRESHOLD MODEL (Extreme Divergence Regimes)")
    print("=" * 70)

    data = df.copy()
    data["log_return"] = np.log(data["usd_jpy_close"] / data["usd_jpy_close"].shift(1))
    data["div_lag"] = data["Policy_Divergence_Hawk"].shift(1)
    data.dropna(subset=["log_return", "div_lag"], inplace=True)

    # Define divergence regimes using quartiles
    q25 = data["div_lag"].quantile(0.25)
    q75 = data["div_lag"].quantile(0.75)
    median = data["div_lag"].median()

    print(f"\n  Divergence distribution:")
    print(f"    Q25={q25:.4f}  Median={median:.4f}  Q75={q75:.4f}")
    print(f"    Min={data['div_lag'].min():.4f}  Max={data['div_lag'].max():.4f}")

    # Test: mean return in each regime
    regimes = {
        "Strong BoJ Hawk (Q1: div < Q25)": data[data["div_lag"] < q25],
        "Mild BoJ Hawk (Q2: Q25 < div < median)": data[(data["div_lag"] >= q25) & (data["div_lag"] < median)],
        "Mild US Hawk (Q3: median < div < Q75)": data[(data["div_lag"] >= median) & (data["div_lag"] < q75)],
        "Strong US Hawk (Q4: div > Q75)": data[data["div_lag"] >= q75],
    }

    print(f"\n  {'Regime':<45s} {'n':>5s} {'Mean Return':>12s} {'Std':>10s} {'t-stat':>8s} {'p-val':>8s}")
    print("  " + "-" * 90)

    from scipy import stats

    for label, subset in regimes.items():
        returns = subset["log_return"]
        n = len(returns)
        mean_ret = returns.mean()
        std_ret = returns.std()

        # t-test: is the mean return significantly different from zero?
        if n > 1:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
        else:
            t_stat, p_val = np.nan, np.nan

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"  {label:<45s} {n:>5d} {mean_ret:>+12.6f} {std_ret:>10.6f} {t_stat:>+8.3f} {p_val:>8.4f} {sig}")

    # Also test: extreme vs middle
    extreme = data[(data["div_lag"] < q25) | (data["div_lag"] >= q75)]
    middle = data[(data["div_lag"] >= q25) & (data["div_lag"] < q75)]

    # Create dummy variables for non-linear regression
    data["extreme_positive"] = (data["div_lag"] >= q75).astype(float)
    data["extreme_negative"] = (data["div_lag"] < q25).astype(float)

    y = data["log_return"]
    X = sm.add_constant(data[["extreme_positive", "extreme_negative"]])
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"\n  Non-linear (dummy) regression:")
    print(f"    Extreme US Hawk dummy: coef={model.params['extreme_positive']:+.6f}  "
          f"p={model.pvalues['extreme_positive']:.4f}")
    print(f"    Extreme BoJ Hawk dummy: coef={model.params['extreme_negative']:+.6f}  "
          f"p={model.pvalues['extreme_negative']:.4f}")
    print(f"    R²={model.rsquared:.6f}")

    return model


# =============================================================================
# TEST 3: MULTI-LAG FORWARD RETURNS
# =============================================================================

def test_multi_lag(df: pd.DataFrame):
    """Test divergence against 1, 5, 10, and 20-day forward returns."""
    print("\n" + "=" * 70)
    print("TEST 3: MULTI-LAG FORWARD RETURNS")
    print("=" * 70)
    print("  Does today's divergence predict returns over the next N days?")

    data = df.copy()
    lags = [1, 5, 10, 20, 40, 60]

    results = []

    print(f"\n  {'Forward Window':<18s} {'n':>5s} {'Div Coef':>12s} {'t-stat':>8s} {'p-value':>10s} {'R²':>8s} {'Sig':>5s}")
    print("  " + "-" * 75)

    for lag in lags:
        data[f"fwd_return_{lag}d"] = np.log(
            data["usd_jpy_close"].shift(-lag) / data["usd_jpy_close"]
        )

        reg = data[["Policy_Divergence_Hawk", "T10Y2Y", "vix", f"fwd_return_{lag}d"]].dropna()

        if len(reg) < 30:
            continue

        y = reg[f"fwd_return_{lag}d"]
        X = sm.add_constant(reg[["Policy_Divergence_Hawk", "T10Y2Y", "vix"]])
        model = sm.OLS(y, X).fit(cov_type="HC1")

        div_coef = model.params["Policy_Divergence_Hawk"]
        div_pval = model.pvalues["Policy_Divergence_Hawk"]
        div_tstat = model.tvalues["Policy_Divergence_Hawk"]
        r2 = model.rsquared

        sig = "***" if div_pval < 0.01 else "**" if div_pval < 0.05 else "*" if div_pval < 0.10 else ""

        results.append({
            "lag": lag,
            "n": int(model.nobs),
            "div_coef": div_coef,
            "div_tstat": div_tstat,
            "div_pval": div_pval,
            "r2": r2,
            "sig": sig,
        })

        print(f"  {lag:>3d}-day forward   {int(model.nobs):>5d} {div_coef:>+12.6f} {div_tstat:>+8.3f} {div_pval:>10.4f} {r2:>8.4f} {sig:>5s}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_robustness_plot(df, multi_lag_results, output_path):
    """4-panel visualization summarizing all robustness tests."""
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

    data = df.copy()
    data["log_return"] = np.log(data["usd_jpy_close"] / data["usd_jpy_close"].shift(1))
    data["div_lag"] = data["Policy_Divergence_Hawk"].shift(1)
    data.dropna(subset=["log_return", "div_lag"], inplace=True)

    # --- Panel 1: Time series of divergence with regime shading ---
    ax1 = axes[0, 0]
    ax1.plot(data.index, data["Policy_Divergence_Hawk"], color="#ff6b6b", linewidth=0.8, alpha=0.7)
    ax1.axhline(0, color="#636e80", linewidth=0.5, linestyle="--")
    ax1.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"),
                alpha=0.1, color="#50fa7b", label="Active Regime (2022-2024)")
    ax1.set_title("Policy Divergence Over Time", fontsize=11, fontweight="bold", color="#ffffff")
    ax1.set_ylabel("US_Hawk - BoJ_Hawk")
    ax1.legend(fontsize=8, framealpha=0.2)
    ax1.grid(True, linewidth=0.3)

    # --- Panel 2: Scatter by regime ---
    ax2 = axes[0, 1]
    pre = data[data.index < "2022-01-01"]
    post = data[data.index >= "2022-01-01"]
    ax2.scatter(pre["div_lag"], pre["log_return"], alpha=0.2, s=8, color="#636e80", label="2016-2021")
    ax2.scatter(post["div_lag"], post["log_return"], alpha=0.35, s=12, color="#ff6b6b", label="2022-2026")

    if len(post) > 5:
        X_post = sm.add_constant(post["div_lag"])
        m = sm.OLS(post["log_return"], X_post).fit()
        x_line = np.linspace(post["div_lag"].min(), post["div_lag"].max(), 50)
        ax2.plot(x_line, m.params["const"] + m.params["div_lag"] * x_line,
                 color="#50fa7b", linewidth=2, label=f"2022+ fit (p={m.pvalues['div_lag']:.3f})")

    ax2.axhline(0, color="#636e80", linewidth=0.5, linestyle="--")
    ax2.axvline(0, color="#636e80", linewidth=0.5, linestyle="--")
    ax2.set_title("Scatter: Divergence vs Daily Return by Regime", fontsize=11, fontweight="bold", color="#ffffff")
    ax2.set_xlabel("Lagged Policy Divergence")
    ax2.set_ylabel("Daily Log Return")
    ax2.legend(fontsize=8, framealpha=0.2)
    ax2.grid(True, linewidth=0.3)

    # --- Panel 3: Multi-lag bar chart ---
    ax3 = axes[1, 0]
    if multi_lag_results:
        lags = [r["lag"] for r in multi_lag_results]
        pvals = [r["div_pval"] for r in multi_lag_results]
        colors = ["#50fa7b" if p < 0.05 else "#ffb86c" if p < 0.10 else "#636e80" for p in pvals]

        bars = ax3.bar(range(len(lags)), pvals, color=colors, edgecolor="#1e2a3a", width=0.6)
        ax3.set_xticks(range(len(lags)))
        ax3.set_xticklabels([f"{l}d" for l in lags])
        ax3.axhline(0.05, color="#50fa7b", linewidth=1, linestyle="--", alpha=0.7, label="5% threshold")
        ax3.axhline(0.10, color="#ffb86c", linewidth=1, linestyle="--", alpha=0.5, label="10% threshold")
        ax3.set_title("P-values by Forward Return Horizon", fontsize=11, fontweight="bold", color="#ffffff")
        ax3.set_xlabel("Forward Return Window")
        ax3.set_ylabel("P-value (lower = more significant)")
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=8, framealpha=0.2)
        ax3.grid(True, axis="y", linewidth=0.3)

    # --- Panel 4: Coefficient magnitude by lag ---
    ax4 = axes[1, 1]
    if multi_lag_results:
        coefs = [r["div_coef"] for r in multi_lag_results]
        colors_coef = ["#ff6b6b" if c < 0 else "#50fa7b" for c in coefs]
        ax4.bar(range(len(lags)), coefs, color=colors_coef, edgecolor="#1e2a3a", width=0.6)
        ax4.set_xticks(range(len(lags)))
        ax4.set_xticklabels([f"{l}d" for l in lags])
        ax4.axhline(0, color="#636e80", linewidth=0.5, linestyle="--")
        ax4.set_title("Divergence Coefficient by Forward Horizon", fontsize=11, fontweight="bold", color="#ffffff")
        ax4.set_xlabel("Forward Return Window")
        ax4.set_ylabel("Coefficient (+ = USD strengthens)")
        ax4.grid(True, axis="y", linewidth=0.3)

    fig.suptitle(
        "Robustness Analysis — Policy Divergence × USD/JPY",
        fontsize=15, fontweight="bold", color="#ffffff", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[OK] Robustness plot saved → {output_path}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

def print_final_verdict(subsample_results, multi_lag_results):
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Check if any subsample is significant
    any_subsample_sig = any(r["div_pval"] < 0.10 for r in subsample_results)

    # Check if any lag is significant
    any_lag_sig = any(r["div_pval"] < 0.10 for r in multi_lag_results)

    # Find best lag
    if multi_lag_results:
        best_lag = min(multi_lag_results, key=lambda r: r["div_pval"])
    else:
        best_lag = None

    # Find best subsample
    best_sub = min(subsample_results, key=lambda r: r["div_pval"])

    print(f"""
  Best subsample:  {best_sub['label']}
                   coef={best_sub['div_coef']:+.6f}, p={best_sub['div_pval']:.4f}, n={best_sub['n']}
""")
    if best_lag:
        print(f"""  Best forward lag: {best_lag['lag']}-day
                   coef={best_lag['div_coef']:+.6f}, p={best_lag['div_pval']:.4f}, n={best_lag['n']}
""")

    if any_subsample_sig or any_lag_sig:
        print("""  STATUS: SIGNAL DETECTED (conditionally)
  
  The policy divergence shows predictive power in specific regimes or
  at specific horizons. This is consistent with a macro factor that
  operates episodically rather than continuously.
  
  RECOMMENDATION: Proceed to backtest with regime-conditional signals.
  The strategy should only be active when divergence exceeds a threshold
  and should target the horizon where significance is strongest.""")

    else:
        print("""  STATUS: NO SIGNIFICANT SIGNAL DETECTED
  
  After testing:
    - Multiple time frequencies (daily, weekly, monthly)
    - Multiple subsamples (full, pre-2022, post-2022, peak divergence)
    - Non-linear threshold models
    - Multiple forward horizons (1d through 60d)
  
  The Fed-BoJ hawkish policy divergence does NOT show statistically
  significant predictive power for USD/JPY returns.
  
  POSSIBLE EXPLANATIONS:
    1. The keyword scorer, while directionally accurate on known events,
       may not capture the nuance that moves markets (tone, surprise
       component, deviation from expectations).
    2. Policy divergence is already priced in by the time documents are
       published — the market moves on expectations, not on the statement.
    3. The relationship may exist at the level of CHANGES in divergence
       rather than levels — a momentum/acceleration signal.
    4. Other macro factors (carry trade flows, risk appetite, intervention)
       may dominate the FX dynamics beyond what sentiment captures.
  
  RECOMMENDATION: 
    - Test divergence CHANGES (first difference) instead of levels
    - Incorporate forward rate expectations (OIS curves) as a control
    - Consider an event study approach around meeting dates only
    - Explore if the signal works for other currency pairs (EUR/USD, GBP/USD)""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ROBUSTNESS & REGIME ANALYSIS")
    print("Final investigation of the Policy Divergence signal")
    print("=" * 70)

    df = load_data()

    # Test 1: Subsample
    sub_results = test_subsample(df)

    # Test 2: Threshold
    threshold_model = test_threshold(df)

    # Test 3: Multi-lag
    lag_results = test_multi_lag(df)

    # Visualization
    print("\n[4] Generating robustness visualization...")
    create_robustness_plot(df, lag_results, PLOT_OUTPUT)

    # Final verdict
    print_final_verdict(sub_results, lag_results)

    print(f"\n{'=' * 70}")
    print("[DONE] Robustness analysis complete.")
    print(f"{'=' * 70}")