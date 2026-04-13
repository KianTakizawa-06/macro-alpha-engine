"""
================================================================================
MONETARY POLICY KEYWORD SCORER
================================================================================
Replaces FinBERT-tone with a domain-specific keyword scorer designed for
central bank statements. Scores each document on hawkish/dovish dimensions
using curated phrase lists derived from actual Fed and BoJ policy language.

WHY KEYWORDS OVER FINBERT?
  FinBERT was trained on financial news headlines (e.g., "Stock surges 10%").
  Central bank statements use institutional hedging language like "the Committee
  decided to raise the target range" — which FinBERT reads as neutral.
  A keyword approach tuned to monetary policy vocabulary captures the actual
  policy signals that FinBERT misses.

SCORING METHOD:
  For each document, we count weighted phrase matches for hawkish and dovish
  terms, then normalize to [0, 1] probabilities. Phrases are weighted by
  signal strength (e.g., "raise rates" is stronger than "monitoring inflation").

Save as:  keyword_scorer.py
Run from: your macro_engine directory
================================================================================
"""

import sqlite3
import re

DB_PATH = "macro_engine.db"


# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================
# Weight scale: 3 = strong signal, 2 = moderate, 1 = weak/contextual
# Phrases are lowercased for matching. Longer phrases are checked first
# to avoid partial matches.

HAWKISH_PHRASES = {
    # --- Direct policy action (strongest signals) ---
    "decided to raise": 3,
    "raise the target range": 3,
    "increase the target range": 3,
    "rate hike": 3,
    "rate increase": 3,
    "raised interest rates": 3,
    "tightening monetary policy": 3,
    "further tightening": 3,
    "additional firming": 3,
    "reducing the size of": 2,          # balance sheet reduction
    "runoff of": 2,                      # balance sheet runoff
    "quantitative tightening": 3,
    "ended negative interest": 3,        # BoJ specific
    "exited negative": 3,
    "widened the band": 2,               # YCC adjustment (BoJ)
    "yield curve control adjustment": 2,

    # --- BoJ-specific hawkish language ---
    # BoJ wraps hawkish actions in hedged language. These phrases
    # specifically capture BoJ policy normalization signals.
    "raise the policy interest rate": 3,
    "raised the policy interest rate": 3,
    "increase the policy interest rate": 3,
    "adjust the degree of monetary accommodation": 3,
    "adjust the degree of monetary easing": 3,
    "reduction of the amount of": 2,          # JGB purchase reduction
    "reduce its purchases": 2,
    "greater flexibility": 2,                  # YCC flexibility = hawkish
    "modify the conduct": 2,
    "modify its yield curve control": 3,
    "discontinue": 2,
    "phase out": 2,
    "normalize": 3,
    "normalization": 3,
    "positive interest rate": 3,               # ending NIRP
    "exit from": 2,
    "virtuous cycle between wages and prices": 2,  # BoJ justification for hikes
    "wage increases": 2,
    "wage-price spiral": 2,
    "sustainable and stable manner": 1,        # BoJ inflation goal language
    "outlook will be realized": 1,             # BoJ forward guidance for hikes
    "not miss the timing": 2,                  # BoJ urgency signal
    "behind the curve": 2,
    "cost of waiting": 2,

    # --- Inflation concern language ---
    "inflation remains elevated": 2,
    "inflation is elevated": 2,
    "inflation has been elevated": 2,
    "inflation remains high": 2,
    "inflation is high": 2,
    "price stability": 1,
    "above 2 percent": 2,
    "above the target": 2,
    "exceeds the target": 2,
    "upside risks to inflation": 2,
    "inflation expectations have risen": 2,
    "wage growth has been strong": 2,
    "wages have increased": 1,
    "cost pressures": 1,
    "price pressures": 1,
    "inflationary pressures": 2,

    # --- Forward guidance (hawkish lean) ---
    "highly attentive to inflation": 2,
    "committed to returning inflation": 2,
    "strongly committed": 2,
    "further increases": 2,
    "ongoing increases": 2,
    "continued increases": 2,
    "additional increases": 2,
    "sufficiently restrictive": 2,
    "maintaining a restrictive": 2,
    "restrictive stance": 2,
    "some further firming": 1,
    "not yet sufficiently restrictive": 2,
    "premature to": 1,                   # "premature to cut"

    # --- Labor market strength (supports hawkish) ---
    "labor market remains tight": 1,
    "labor market is tight": 1,
    "strong labor market": 1,
    "robust job gains": 1,
    "job gains have been strong": 1,
    "solid pace of job gains": 1,
    "unemployment remains low": 1,
    "unemployment rate has remained low": 1,
}

DOVISH_PHRASES = {
    # --- Direct policy action (strongest signals) ---
    "decided to lower": 3,
    "lower the target range": 3,
    "decrease the target range": 3,
    "rate cut": 3,
    "rate reduction": 3,
    "lowered interest rates": 3,
    "easing monetary policy": 3,
    "policy easing": 3,
    "accommodation": 1,
    "accommodative financial conditions": 1,
    "maintain the current target": 1,
    "decided to maintain": 1,
    "asset purchases": 2,
    "quantitative easing": 3,
    "increase its holdings": 2,
    "reinvesting": 1,
    "applying a negative interest rate": 2,   # Specifically maintaining NIRP

    # --- Inflation softening language ---
    "inflation has eased": 2,
    "inflation has declined": 2,
    "inflation has come down": 2,
    "inflation has moderated": 2,
    "inflation has made progress": 2,
    "considerable progress": 2,
    "further progress": 1,
    "moving toward": 1,
    "closer to the target": 1,
    "toward 2 percent": 1,
    "below the target": 2,
    "below 2 percent": 2,
    "downside risks to inflation": 2,
    "disinflation": 2,
    "deflationary": 2,
    "deflation": 2,

    # --- Forward guidance (dovish lean) ---
    "prepared to adjust": 1,
    "appropriate to reduce": 2,
    "appropriate to lower": 2,
    "gained greater confidence": 2,
    "gaining confidence": 1,
    "well positioned to": 1,
    "patient": 1,
    "gradual": 1,
    "data dependent": 1,
    "carefully assess": 1,
    "closely monitoring": 1,

    # --- Economic weakness signals ---
    "economic activity has slowed": 2,
    "growth has slowed": 2,
    "slowdown": 1,
    "recession": 2,
    "contraction": 2,
    "economic weakness": 2,
    "labor market has softened": 2,
    "job gains have slowed": 1,
    "unemployment has risen": 2,
    "unemployment rate has moved up": 1,
    "financial stress": 2,
    "banking sector stress": 2,
    "tighter credit conditions": 2,
    "credit conditions have tightened": 2,
    "downside risks": 1,
    "uncertainty": 1,
}


# =============================================================================
# SCORING ENGINE
# =============================================================================

def score_document(text: str) -> tuple:
    """
    Score a document's hawkish and dovish content using weighted keyword matching.
    
    Returns: (hawkish_score, dovish_score, neutral_score) normalized to sum to 1.0
    """
    if not text or len(text.strip()) == 0:
        return 0.0, 0.0, 1.0

    text_lower = text.lower()

    # Count weighted matches
    hawk_weight = 0.0
    dove_weight = 0.0

    for phrase, weight in HAWKISH_PHRASES.items():
        # Count all occurrences of the phrase
        count = len(re.findall(re.escape(phrase), text_lower))
        hawk_weight += count * weight

    for phrase, weight in DOVISH_PHRASES.items():
        count = len(re.findall(re.escape(phrase), text_lower))
        dove_weight += count * weight

    total_weight = hawk_weight + dove_weight

    if total_weight == 0:
        # No policy signals detected
        return 0.0, 0.0, 1.0

    # Normalize: convert raw weights to probabilities
    # We scale so hawkish + dovish + neutral = 1.0
    # Neutral represents the portion of the document without clear signals
    raw_hawk = hawk_weight / total_weight
    raw_dove = dove_weight / total_weight

    # Scale down based on signal density (more matches = more confident)
    # A document with 1 match should still be mostly neutral
    # A document with 20+ matches should have strong hawk/dove scores
    confidence = min(total_weight / 30.0, 1.0)  # saturates at 30 weighted points

    hawkish_score = raw_hawk * confidence
    dovish_score = raw_dove * confidence
    neutral_score = 1.0 - hawkish_score - dovish_score

    return (
        round(float(hawkish_score), 6),
        round(float(dovish_score), 6),
        round(float(neutral_score), 6),
    )


# =============================================================================
# DATABASE UPDATE
# =============================================================================

def rescore_all_documents():
    """Re-score every document in the database with the keyword scorer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load all documents
    cursor.execute("SELECT id, date, source, text_content FROM text_data")
    rows = cursor.fetchall()

    print(f"Re-scoring {len(rows)} documents with keyword scorer...\n")

    results = []
    for row in rows:
        doc_id, date, source, text = row
        hawk, dove, neut = score_document(text)
        results.append((hawk, dove, neut, doc_id))

        # Show what we're scoring
        signal = "HAWKISH" if hawk > dove else "DOVISH" if dove > hawk else "NEUTRAL"
        print(f"  {date} | {source:20s} | H={hawk:.4f} D={dove:.4f} N={neut:.4f} | {signal}")

    # Batch update
    cursor.executemany(
        "UPDATE text_data SET hawkish_score = ?, dovish_score = ?, neutral_score = ? WHERE id = ?",
        results,
    )
    conn.commit()

    # Summary
    print(f"\n{'=' * 60}")
    print("RESCORING SUMMARY")
    print(f"{'=' * 60}")

    cursor.execute("""
        SELECT source, 
               AVG(hawkish_score) as avg_hawk,
               AVG(dovish_score) as avg_dove,
               AVG(neutral_score) as avg_neut,
               COUNT(*) as n
        FROM text_data 
        GROUP BY source
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:20s}  avg_hawk={row[1]:.4f}  avg_dove={row[2]:.4f}  "
              f"avg_neut={row[3]:.4f}  (n={row[4]})")

    conn.close()
    print(f"\n[DONE] All {len(rows)} documents rescored.")
    print("Next: re-run macro_convergence.py → diagnostic_sentiment.py → macro_alpha_poc.py")


if __name__ == "__main__":
    rescore_all_documents()