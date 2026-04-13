"""
================================================================================
FED DATA GAP FILLER
================================================================================
Scrapes FOMC statements from Aug 2023 → present to fill the gap in macro_engine.db.
Uses both the historical archive pages AND the calendar/press release pages
to catch all statements regardless of how the Fed organizes them.

Save as:  fill_fed_gap.py
Run from: your macro_engine directory
================================================================================
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re

DB_PATH = "macro_engine.db"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE_URL = "https://www.federalreserve.gov"


# =============================================================================
# STEP 1: Find all FOMC statement URLs we're missing
# =============================================================================

def get_statement_links(start_year, end_year):
    """
    Scrape FOMC statement links from both:
      - Historical archive pages (fomchistorical{year}.htm)
      - Calendar pages (fomccalendars.htm and fomcpresconf{year}.htm)
    """
    statement_urls = []

    # --- Source A: Historical archives ---
    for year in range(start_year, end_year + 1):
        print(f"  Scanning historical archive for {year}...")
        archive_url = f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"

        try:
            response = requests.get(archive_url, headers=HEADERS)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if "newsevents/pressreleases/monetary" in href and "a.htm" in href:
                        full_url = BASE_URL + href if href.startswith("/") else href
                        if full_url not in statement_urls:
                            statement_urls.append(full_url)
            time.sleep(1)
        except Exception as e:
            print(f"    Error on historical {year}: {e}")

    # --- Source B: Recent calendar/press conference pages ---
    calendar_urls = [
        f"{BASE_URL}/monetarypolicy/fomccalendars.htm",
    ]
    for year in range(start_year, end_year + 1):
        calendar_urls.append(f"{BASE_URL}/monetarypolicy/fomcpresconf{year}.htm")

    for cal_url in calendar_urls:
        print(f"  Scanning calendar page: {cal_url}...")
        try:
            response = requests.get(cal_url, headers=HEADERS)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if "newsevents/pressreleases/monetary" in href and "a.htm" in href:
                        full_url = BASE_URL + href if href.startswith("/") else href
                        if full_url not in statement_urls:
                            statement_urls.append(full_url)
            time.sleep(1)
        except Exception as e:
            print(f"    Error on calendar page: {e}")

    print(f"\n  Total unique statement URLs found: {len(statement_urls)}")
    return statement_urls


# =============================================================================
# STEP 2: Extract date from URL
# =============================================================================

def extract_date_from_url(url):
    """
    Fed statement URLs look like:
      .../monetary20230726a.htm
    Extract the date as YYYY-MM-DD.
    """
    match = re.search(r"monetary(\d{4})(\d{2})(\d{2})a\.htm", url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


# =============================================================================
# STEP 3: Scrape and store, skipping duplicates
# =============================================================================

def scrape_and_store(urls, conn):
    """Download each statement, extract text, insert if not already in DB."""
    cursor = conn.cursor()

    # Get existing dates to skip
    cursor.execute(
        "SELECT date FROM text_data WHERE source = 'Federal Reserve'"
    )
    existing_dates = {row[0] for row in cursor.fetchall()}
    print(f"  Already in DB: {len(existing_dates)} Fed statements")

    added = 0
    skipped = 0

    for url in urls:
        date_str = extract_date_from_url(url)
        if not date_str:
            print(f"  Could not parse date from: {url}")
            continue

        if date_str in existing_dates:
            skipped += 1
            continue

        print(f"  Scraping {date_str}...", end=" ")

        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            article_div = soup.find("div", id="article")
            if not article_div:
                # Fallback: try common alternative containers
                article_div = soup.find("div", class_="col-xs-12")

            if article_div:
                raw_text = article_div.get_text(separator=" ", strip=True)
                raw_text = re.sub(r"\s+", " ", raw_text).strip()

                cursor.execute(
                    """INSERT INTO text_data (date, source, doc_type, text_content)
                       VALUES (?, ?, ?, ?)""",
                    (date_str, "Federal Reserve", "Statement", raw_text),
                )
                conn.commit()
                added += 1
                print(f"OK ({len(raw_text)} chars)")
            else:
                print("WARN: no article div found")

            time.sleep(1.5)

        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n  Done! Added {added} new statements, skipped {skipped} already in DB.")
    return added


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FED DATA GAP FILLER")
    print("Filling FOMC statements from 2023 → 2026")
    print("=" * 60)

    # Search 2023-2026 to catch everything we're missing
    print("\n[1] Finding statement URLs...")
    urls = get_statement_links(2023, 2026)

    if not urls:
        print("No URLs found. Check your internet connection.")
    else:
        print(f"\n[2] Scraping and storing new statements...")
        conn = sqlite3.connect(DB_PATH)
        added = scrape_and_store(urls, conn)

        # Verify final state
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM text_data WHERE source = 'Federal Reserve'"
        )
        row = cursor.fetchone()
        print(f"\n[3] Fed coverage now: {row[0]} → {row[1]} ({row[2]} statements)")

        conn.close()

    print("\n[DONE] Next step: re-run the sentiment scoring pipeline on the new documents.")