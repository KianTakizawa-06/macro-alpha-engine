"""
================================================================================
BOJ DATA GAP FILLER
================================================================================
Fills the 2018-2025 gap in BoJ Summary of Opinions documents.

The issue with the original scraper: from ~2018 onward, the BoJ index pages
link to .htm pages (e.g., opi221220.htm) instead of directly to .pdf files.
The original scraper only looked for links ending in .pdf, so it missed
everything after 2017.

This script handles BOTH formats:
  - Old format (2016-2017): index page → direct .pdf link
  - New format (2018+): index page → .htm page → .pdf link inside that page

Save as:  fill_boj_gap.py
Run from: your macro_engine directory
================================================================================
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import fitz  # PyMuPDF

DB_PATH = "macro_engine.db"
BASE_URL = "https://www.boj.or.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# =============================================================================
# STEP 1: Crawl index pages for all years
# =============================================================================

def get_opinion_links(start_year, end_year):
    """
    Crawl BoJ opinion index pages and collect links to opinion pages/PDFs.
    Returns a list of dicts with 'url', 'date', and 'type' (htm or pdf).
    """
    all_links = []

    for year in range(start_year, end_year + 1):
        archive_url = f"{BASE_URL}/en/mopo/mpmsche_minu/opinion_{year}/index.htm"
        print(f"  Scanning {year}: {archive_url}")

        try:
            response = requests.get(archive_url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"    Status {response.status_code}, skipping")
                continue

            soup = BeautifulSoup(response.content, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Match opinion links: both .htm and .pdf formats
                # Pattern: opi{YYMMDD}.htm or opi{YYMMDD}.pdf
                if re.search(r"opi\d{6}\.(htm|pdf)", href):
                    if href.startswith("/"):
                        full_url = BASE_URL + href
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = f"{BASE_URL}/en/mopo/mpmsche_minu/opinion_{year}/{href}"

                    # Extract date from URL
                    date_match = re.search(r"opi(\d{2})(\d{2})(\d{2})", href)
                    if date_match:
                        yy, mm, dd = date_match.groups()
                        date_str = f"20{yy}-{mm}-{dd}"
                    else:
                        date_str = "Unknown"

                    file_type = "pdf" if href.endswith(".pdf") else "htm"

                    if full_url not in [l["url"] for l in all_links]:
                        all_links.append({
                            "url": full_url,
                            "date": date_str,
                            "type": file_type,
                        })

            time.sleep(1)

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n  Total opinion links found: {len(all_links)}")
    return all_links


# =============================================================================
# STEP 2: For .htm links, find the actual PDF URL inside the page
# =============================================================================

def resolve_pdf_url(htm_url):
    """
    Given an .htm opinion page, find the PDF link within it.
    BoJ opinion .htm pages typically contain a link to the full PDF.
    """
    try:
        response = requests.get(htm_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # Look for PDF links on the page
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") and "opi" in href:
                if href.startswith("/"):
                    return BASE_URL + href
                elif href.startswith("http"):
                    return href
                else:
                    # Relative URL — construct from the htm page's directory
                    base_dir = htm_url.rsplit("/", 1)[0]
                    return base_dir + "/" + href

        # If no opi PDF found, look for any PDF link
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf"):
                if href.startswith("/"):
                    return BASE_URL + href
                elif href.startswith("http"):
                    return href

        # Some pages have the content inline — extract text directly
        return None

    except Exception as e:
        print(f"    Error resolving PDF from {htm_url}: {e}")
        return None


def extract_text_from_htm(htm_url):
    """
    Fallback: if no PDF is found, extract text directly from the .htm page.
    Some BoJ opinion pages have the full text in HTML format.
    """
    try:
        response = requests.get(htm_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # The main content is usually in a div with id "contents" or class "content"
        content_div = (
            soup.find("div", id="contents")
            or soup.find("div", class_="content")
            or soup.find("div", id="main")
        )

        if content_div:
            text = content_div.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 500:  # Sanity check — real content should be substantial
                return text

        return None

    except Exception as e:
        print(f"    Error extracting HTM text from {htm_url}: {e}")
        return None


# =============================================================================
# STEP 3: Download PDF and extract text
# =============================================================================

def extract_text_from_pdf(pdf_url):
    """Download PDF bytes and extract text using PyMuPDF."""
    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        full_text = ""
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            full_text += page.get_text("text") + " "
        pdf_doc.close()

        clean_text = re.sub(r"\s+", " ", full_text).strip()
        return clean_text

    except Exception as e:
        print(f"    Error extracting PDF text: {e}")
        return None


# =============================================================================
# STEP 4: Store in database
# =============================================================================

def store_and_score(links, conn):
    """Process each link, extract text, store in DB, skip duplicates."""
    cursor = conn.cursor()

    # Get existing BoJ dates
    cursor.execute(
        "SELECT date FROM text_data WHERE source = 'Bank of Japan'"
    )
    existing_dates = {row[0] for row in cursor.fetchall()}
    print(f"  Already in DB: {len(existing_dates)} BoJ documents")

    added = 0
    skipped = 0
    failed = 0

    for item in links:
        date_str = item["date"]
        url = item["url"]
        link_type = item["type"]

        if date_str in existing_dates:
            skipped += 1
            continue

        print(f"  Processing {date_str} ({link_type})...", end=" ")

        text = None

        if link_type == "pdf":
            # Direct PDF link
            text = extract_text_from_pdf(url)

        elif link_type == "htm":
            # First try to find a PDF link inside the .htm page
            pdf_url = resolve_pdf_url(url)
            if pdf_url:
                print(f"→ PDF: {pdf_url.split('/')[-1]}", end=" ")
                text = extract_text_from_pdf(pdf_url)

            # Fallback: extract text from the HTML page itself
            if not text:
                print("→ HTM fallback", end=" ")
                text = extract_text_from_htm(url)

        if text and len(text) > 500:
            cursor.execute(
                """INSERT INTO text_data (date, source, doc_type, text_content)
                   VALUES (?, ?, ?, ?)""",
                (date_str, "Bank of Japan", "Summary of Opinions", text),
            )
            conn.commit()
            added += 1
            print(f"OK ({len(text)} chars)")
        else:
            failed += 1
            print("FAILED (no text extracted)")

        time.sleep(2)  # Be polite to BoJ servers

    print(f"\n  Done! Added: {added}, Skipped: {skipped}, Failed: {failed}")
    return added


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BOJ DATA GAP FILLER")
    print("Filling BoJ Summary of Opinions 2018 → 2025")
    print("=" * 60)

    print("\n[1] Finding opinion links...")
    links = get_opinion_links(2018, 2025)

    if not links:
        print("No links found. Check your internet connection.")
    else:
        print(f"\n[2] Processing and storing documents...")
        conn = sqlite3.connect(DB_PATH)
        added = store_and_score(links, conn)

        # Verify
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM text_data WHERE source = 'Bank of Japan'"
        )
        row = cursor.fetchone()
        print(f"\n[3] BoJ coverage now: {row[0]} → {row[1]} ({row[2]} documents)")

        conn.close()

    print("\n[DONE] Next steps:")
    print("  1. python3 keyword_scorer.py    (re-score all documents)")
    print("  2. python3 macro_convergence.py  (rebuild dataset)")
    print("  3. python3 diagnostic_sentiment.py (validate)")
    print("  4. python3 multi_freq_regression.py (re-run analysis)")