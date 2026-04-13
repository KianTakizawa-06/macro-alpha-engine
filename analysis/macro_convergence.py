import sqlite3

conn = sqlite3.connect('macro_engine.db')
cursor = conn.cursor()

# Check the table schema
cursor.execute("PRAGMA table_info(text_data)")
print("Schema:")
for row in cursor.fetchall():
    print(f"  {row}")

# Look at raw values
cursor.execute("SELECT date, source, hawkish_score, dovish_score FROM text_data LIMIT 5")
print("\nRaw sample rows:")
for row in cursor.fetchall():
    print(f"  date={row[0]}, source={row[1]}")
    print(f"    hawkish_score: type={type(row[2])}, value={repr(row[2])}")
    print(f"    dovish_score:  type={type(row[3])}, value={repr(row[3])}")

# Check if there are ANY plain numeric values
cursor.execute("""
    SELECT COUNT(*) FROM text_data 
    WHERE typeof(hawkish_score) IN ('real', 'integer')
""")
print(f"\nRows with numeric hawkish_score: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM text_data WHERE hawkish_score IS NOT NULL")
print(f"Total non-null rows: {cursor.fetchone()[0]}")

conn.close()


import pandas as pd
import sqlite3
import sys
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

MARKET_CSV = "market_data_benchmark.csv"
SQLITE_DB = "macro_engine.db"
OUTPUT_CSV = "master_alpha_dataset.csv"

# Map raw source labels from the database to standardized short names.
# Update these keys to match whatever your NLP pipeline stored.
SOURCE_NAME_MAP = {
    "Federal Reserve": "US",
    "Bank of Japan": "BoJ",
    # Common variants — add yours if different:
    # "Fed": "US",
    # "BOJ": "BoJ",
    # "fed": "US",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data(csv_path: str) -> pd.DataFrame:
    """Load daily market CSV and set a DatetimeIndex."""
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"[FATAL] Market CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Date" not in df.columns and "date" not in df.columns:
        sys.exit(f"[FATAL] No 'Date' column found in {csv_path}. Columns: {df.columns.tolist()}")

    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    print(f"[OK] Market data loaded: {df.shape[0]} trading days, {df.shape[1]} features")
    print(f"     Range: {df.index.min().date()} → {df.index.max().date()}")
    return df


def load_sentiment_data(db_path: str) -> pd.DataFrame:
    """Load NLP sentiment scores from SQLite, cast to numeric, normalize source names."""
    path = Path(db_path)
    if not path.exists():
        sys.exit(f"[FATAL] SQLite database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    query = """
        SELECT date, source, hawkish_score, dovish_score
        FROM text_data
        WHERE hawkish_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        sys.exit("[FATAL] No scored rows returned from text_data table.")

    # --- Fix dtype: scores may be stored as TEXT in SQLite ---
    df["hawkish_score"] = pd.to_numeric(df["hawkish_score"], errors="coerce")
    df["dovish_score"] = pd.to_numeric(df["dovish_score"], errors="coerce")

    # Report and drop corrupt rows
    bad_mask = df["hawkish_score"].isna() | df["dovish_score"].isna()
    if bad_mask.any():
        n_bad = bad_mask.sum()
        print(f"[WARN] {n_bad} rows had non-numeric scores — dropped:")
        print(df.loc[bad_mask, ["date", "source"]].head(10).to_string(index=False))
        df = df[~bad_mask].copy()

    df["date"] = pd.to_datetime(df["date"])

    # --- Normalize source names ---
    raw_sources = df["source"].unique()
    print(f"[INFO] Raw sources in DB: {raw_sources}")

    df["source"] = df["source"].map(SOURCE_NAME_MAP).fillna(df["source"])
    mapped_sources = df["source"].unique()
    print(f"[INFO] Mapped sources:    {mapped_sources}")

    if len(mapped_sources) < 2:
        print(f"[WARN] Expected 2 sources (US, BoJ), found {len(mapped_sources)}: {mapped_sources}")
        print("       Update SOURCE_NAME_MAP in the config section to match your DB values.")

    print(f"[OK] Sentiment data loaded: {df.shape[0]} scored documents")
    return df


# =============================================================================
# PIVOT & MERGE
# =============================================================================

def pivot_sentiment(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot sparse event data into a wide-format DataFrame:
    one row per date, columns like US_Hawkish, BoJ_Dovish, etc.

    If two documents from the same source land on the same day,
    we average them — this is conservative and avoids duplication.
    """
    pivot = text_df.pivot_table(
        index="date",
        columns="source",
        values=["hawkish_score", "dovish_score"],
        aggfunc="mean",
    )

    # Flatten MultiIndex columns: ('hawkish_score', 'US') → 'US_Hawkish'
    pivot.columns = [
        f"{src}_{score.split('_')[0].title()}" for score, src in pivot.columns
    ]

    pivot.index.name = "Date"
    pivot.sort_index(inplace=True)

    print(f"[OK] Pivoted sentiment columns: {pivot.columns.tolist()}")
    return pivot


def merge_and_fill(market_df: pd.DataFrame, pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sentiment onto the daily market calendar, then forward-fill.

    WHY left join + ffill?
    - The market calendar is our authoritative time axis (252 days/yr).
    - Sentiment events are sparse (~16 per year across both banks).
    - left join places each event on its publication date.
    - ffill propagates the "regime" forward to every subsequent trading day
      WITHOUT any look-ahead bias — only past values are carried forward.
    """
    master = market_df.join(pivot_df, how="left")

    # Identify all sentiment columns dynamically
    sentiment_cols = [c for c in master.columns if "Hawkish" in c or "Dovish" in c]

    if not sentiment_cols:
        sys.exit("[FATAL] No sentiment columns found after merge. Check pivot output.")

    # Forward-fill: carry the last known regime into future trading days
    master[sentiment_cols] = master[sentiment_cols].ffill()

    # Drop the initial window before the first central bank event
    # (these rows have no regime information and would introduce NaN bias)
    n_before = len(master)
    master.dropna(subset=sentiment_cols, inplace=True)
    n_dropped = n_before - len(master)

    print(f"[OK] Merged dataset: {len(master)} rows ({n_dropped} pre-regime rows dropped)")
    return master


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build alpha features from the merged dataset.

    Policy_Divergence_Hawk:
        US_Hawkish - BoJ_Hawkish
        Positive → Fed more hawkish than BoJ → higher rate differential → USD strength
        Negative → BoJ more hawkish → JPY strength

    Policy_Divergence_Dove:
        US_Dovish - BoJ_Dovish
        Positive → Fed more dovish → USD weakness signal

    T10Y2Y:
        US 10Y yield minus 2Y yield (term spread / recession indicator)
    """
    # Yield curve spread
    if "yield_10y" in df.columns and "yield_2y" in df.columns:
        df["T10Y2Y"] = df["yield_10y"] - df["yield_2y"]
        print("[OK] Engineered: T10Y2Y (yield curve spread)")

    # Policy divergence — built dynamically from available columns
    hawk_cols = sorted([c for c in df.columns if "Hawkish" in c])
    dove_cols = sorted([c for c in df.columns if "Dovish" in c])

    if len(hawk_cols) == 2:
        # sorted() ensures consistent order: BoJ before US alphabetically
        # So hawk_cols[0] = BoJ_Hawkish, hawk_cols[1] = US_Hawkish
        us_hawk = [c for c in hawk_cols if c.startswith("US")]
        boj_hawk = [c for c in hawk_cols if c.startswith("BoJ")]

        if us_hawk and boj_hawk:
            df["Policy_Divergence_Hawk"] = df[us_hawk[0]] - df[boj_hawk[0]]
            print(f"[OK] Engineered: Policy_Divergence_Hawk = {us_hawk[0]} - {boj_hawk[0]}")
        else:
            print(f"[WARN] Could not identify US/BoJ hawkish columns from: {hawk_cols}")
    else:
        print(f"[WARN] Expected 2 Hawkish columns, found {len(hawk_cols)}: {hawk_cols}")

    if len(dove_cols) == 2:
        us_dove = [c for c in dove_cols if c.startswith("US")]
        boj_dove = [c for c in dove_cols if c.startswith("BoJ")]

        if us_dove and boj_dove:
            df["Policy_Divergence_Dove"] = df[us_dove[0]] - df[boj_dove[0]]
            print(f"[OK] Engineered: Policy_Divergence_Dove = {us_dove[0]} - {boj_dove[0]}")
        else:
            print(f"[WARN] Could not identify US/BoJ dovish columns from: {dove_cols}")
    else:
        print(f"[WARN] Expected 2 Dovish columns, found {len(dove_cols)}: {dove_cols}")

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_master_dataset() -> pd.DataFrame:
    """Execute the full pipeline: load → pivot → merge → engineer → save."""
    print("=" * 70)
    print("MACRO CONVERGENCE ENGINE — Building Master Alpha Dataset")
    print("=" * 70)

    # Step 1: Load both data sources
    market_df = load_market_data(MARKET_CSV)
    text_df = load_sentiment_data(SQLITE_DB)

    # Step 2: Pivot sparse sentiment into wide format
    pivot_df = pivot_sentiment(text_df)

    # Step 3: Merge onto daily calendar with forward-fill
    master_df = merge_and_fill(market_df, pivot_df)

    # Step 4: Engineer alpha features
    master_df = engineer_features(master_df)

    # Step 5: Save
    master_df.to_csv(OUTPUT_CSV)
    print("=" * 70)
    print(f"[DONE] Saved to {OUTPUT_CSV}")
    print(f"       Shape: {master_df.shape[0]} rows × {master_df.shape[1]} columns")
    print(f"       Columns: {master_df.columns.tolist()}")
    print("=" * 70)

    return master_df


if __name__ == "__main__":
    df = build_master_dataset()
    print("\nSample output (last 5 rows):")
    print(df.tail().to_string())