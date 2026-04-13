"""
================================================================================
SENTIMENT VALIDATION DIAGNOSTIC
================================================================================
Validates FinBERT sentiment scores against known monetary policy events
to check whether the scoring pipeline captures hawkish/dovish shifts correctly.

Save this as: diagnostic_sentiment.py
Run from:     your macro_engine directory (same folder as macro_engine.db)
================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "macro_engine.db"


# =============================================================================
# KNOWN POLICY EVENTS (Ground Truth)
# =============================================================================
# These are dates where the policy direction is unambiguous.
# We check whether FinBERT's scores align with what actually happened.

KNOWN_EVENTS = [
    # --- FEDERAL RESERVE ---
    # 2022 tightening cycle — the most aggressively hawkish Fed in decades
    {"date": "2022-03-16", "source_contains": "Federal Reserve",
     "expected": "hawkish", "event": "First rate hike (0→0.25%), signaling tightening cycle"},
    {"date": "2022-06-15", "source_contains": "Federal Reserve",
     "expected": "hawkish", "event": "75bp hike — largest since 1994"},
    {"date": "2022-09-21", "source_contains": "Federal Reserve",
     "expected": "hawkish", "event": "Third consecutive 75bp hike"},
    {"date": "2022-11-02", "source_contains": "Federal Reserve",
     "expected": "hawkish", "event": "Fourth consecutive 75bp hike"},
    # 2023-2024 — holding then pivoting
    {"date": "2023-07-26", "source_contains": "Federal Reserve",
     "expected": "hawkish", "event": "Final hike to 5.25-5.50%"},
    {"date": "2024-09-18", "source_contains": "Federal Reserve",
     "expected": "dovish", "event": "First rate cut (50bp) — pivot to easing"},

    # --- BANK OF JAPAN ---
    # 2022-2023 — ultra-dovish, defending yield curve control
    {"date": "2022-12-20", "source_contains": "Bank of Japan",
     "expected": "hawkish", "event": "YCC band widened — surprise hawkish shift"},
    # 2024 — historic policy shift
    {"date": "2024-03-19", "source_contains": "Bank of Japan",
     "expected": "hawkish", "event": "First rate hike since 2007, ended negative rates"},
    {"date": "2024-07-31", "source_contains": "Bank of Japan",
     "expected": "hawkish", "event": "Second rate hike to 0.25%"},
]


# =============================================================================
# LOAD AND INSPECT
# =============================================================================

def load_all_scores() -> pd.DataFrame:
    """Load all scored documents from the database."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id, date, source, doc_type, hawkish_score, dovish_score, neutral_score,
               LENGTH(text_content) as text_length
        FROM text_data
        WHERE hawkish_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Handle bytes if the repair script hasn't been run
    import struct
    for col in ["hawkish_score", "dovish_score", "neutral_score"]:
        df[col] = df[col].apply(
            lambda v: struct.unpack('<f', v)[0] if isinstance(v, bytes) and len(v) == 4
            else float(v) if v is not None else np.nan
        )

    df["date"] = pd.to_datetime(df["date"])
    return df


def run_diagnostic():
    print("=" * 70)
    print("SENTIMENT VALIDATION DIAGNOSTIC")
    print("=" * 70)

    df = load_all_scores()

    # -----------------------------------------------------------------
    # SECTION 1: Overview of what's in the database
    # -----------------------------------------------------------------
    print("\n[1] DATABASE OVERVIEW")
    print("-" * 50)
    print(f"Total scored documents: {len(df)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nDocuments per source:")
    print(df.groupby("source").size().to_string())
    print(f"\nDocument types:")
    print(df.groupby("doc_type").size().to_string())

    # -----------------------------------------------------------------
    # SECTION 2: Score distribution — are scores meaningful?
    # -----------------------------------------------------------------
    print("\n\n[2] SCORE DISTRIBUTIONS")
    print("-" * 50)
    for source in df["source"].unique():
        subset = df[df["source"] == source]
        print(f"\n  {source} ({len(subset)} docs):")
        for col in ["hawkish_score", "dovish_score", "neutral_score"]:
            vals = subset[col].dropna()
            print(f"    {col:18s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    # Check if neutral dominates (common FinBERT issue with policy text)
    avg_neutral = df["neutral_score"].mean()
    if avg_neutral > 0.7:
        print(f"\n  [WARN] Average neutral score is {avg_neutral:.2f} — FinBERT is classifying")
        print(f"         most policy text as neutral. Hawkish/dovish signals may be too weak")
        print(f"         to be informative. Consider a domain-specific model.")

    # -----------------------------------------------------------------
    # SECTION 3: Validate against known events
    # -----------------------------------------------------------------
    print("\n\n[3] KNOWN EVENT VALIDATION")
    print("-" * 50)

    hits = 0
    misses = 0
    not_found = 0

    for event in KNOWN_EVENTS:
        target_date = pd.to_datetime(event["date"])
        source_filter = event["source_contains"]

        # Search within a 7-day window (meeting dates may not align exactly)
        window_start = target_date - pd.Timedelta(days=3)
        window_end = target_date + pd.Timedelta(days=3)

        matches = df[
            (df["date"] >= window_start) &
            (df["date"] <= window_end) &
            (df["source"].str.contains(source_filter, case=False))
        ]

        print(f"\n  Event: {event['event']}")
        print(f"  Date:  {event['date']}  |  Expected: {event['expected'].upper()}")

        if matches.empty:
            print(f"  Result: ❌ NO DOCUMENT FOUND in ±3 day window")
            not_found += 1
            continue

        # Take the closest match
        row = matches.iloc[0]
        hawk = row["hawkish_score"]
        dove = row["dovish_score"]
        neut = row["neutral_score"]

        # Determine what FinBERT thinks
        scores = {"hawkish": hawk, "dovish": dove, "neutral": neut}
        predicted = max(scores, key=scores.get)

        # Check alignment
        if event["expected"] == "hawkish":
            correct = hawk > dove
        else:
            correct = dove > hawk

        status = "✅ CORRECT" if correct else "❌ WRONG"
        if correct:
            hits += 1
        else:
            misses += 1

        print(f"  Scores: hawk={hawk:.4f}  dove={dove:.4f}  neut={neut:.4f}  → {predicted}")
        print(f"  Result: {status}")

        if not correct:
            print(f"  [ISSUE] Expected {event['expected']} but hawk={hawk:.4f} vs dove={dove:.4f}")

    # -----------------------------------------------------------------
    # SECTION 4: Summary verdict
    # -----------------------------------------------------------------
    total_tested = hits + misses
    accuracy = hits / total_tested * 100 if total_tested > 0 else 0

    print("\n\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Events tested:    {total_tested}")
    print(f"  Correct:          {hits}")
    print(f"  Wrong:            {misses}")
    print(f"  Not in DB:        {not_found}")
    print(f"  Accuracy:         {accuracy:.0f}%")

    if accuracy >= 75:
        print("\n  VERDICT: Scores appear reasonably aligned with known policy shifts.")
        print("           Proceed with the alpha model, but monitor for noise.")
    elif accuracy >= 50:
        print("\n  VERDICT: Mixed results. FinBERT captures some signals but misses others.")
        print("           Consider supplementing with a keyword-based hawkish/dovish scorer")
        print("           or fine-tuning on central bank language.")
    else:
        print("\n  VERDICT: FinBERT-tone is NOT reliably capturing monetary policy sentiment.")
        print("           The model was trained on financial news, not central bank statements.")
        print("           RECOMMENDED ACTIONS:")
        print("             1. Fine-tune FinBERT on labeled central bank text")
        print("             2. Use a keyword/rule-based scorer as a baseline")
        print("             3. Try a model trained on monetary policy (e.g., cb-FinBERT)")

    # -----------------------------------------------------------------
    # SECTION 5: Score over time — look for suspicious patterns
    # -----------------------------------------------------------------
    print("\n\n[5] TEMPORAL PATTERNS")
    print("-" * 50)

    for source in df["source"].unique():
        subset = df[df["source"] == source].sort_values("date")
        if len(subset) < 2:
            continue

        # Check if scores are suspiciously constant
        hawk_std = subset["hawkish_score"].std()
        if hawk_std < 0.01:
            print(f"\n  [WARN] {source}: hawkish_score std = {hawk_std:.6f}")
            print(f"         Scores are nearly constant — model may not be discriminating.")

        # Check for duplicate scores (copy-paste or caching issue)
        dupes = subset.duplicated(subset=["hawkish_score", "dovish_score"], keep=False)
        n_dupes = dupes.sum()
        if n_dupes > len(subset) * 0.3:
            print(f"\n  [WARN] {source}: {n_dupes}/{len(subset)} rows have duplicate scores.")
            print(f"         This suggests the model is producing identical outputs for")
            print(f"         different documents — check chunking and input preprocessing.")

    print("\n[DONE] Diagnostic complete.")


if __name__ == "__main__":
    run_diagnostic()