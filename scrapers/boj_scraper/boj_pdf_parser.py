import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import fitz  # This is PyMuPDF

# 1. Database Connection (Hooks into the same DB as the Fed data!)
def get_db_connection():
    conn = sqlite3.connect('macro_engine.db')
    return conn

# 2. Crawler: Find the PDF Links
def get_boj_pdf_links(start_year, end_year):
    base_url = "https://www.boj.or.jp"
    pdf_urls = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    print(f"Crawling BoJ archives from {start_year} to {end_year}...")
    
    # The BoJ lists their summaries by year
    for year in range(start_year, end_year + 1):
        # Note: Depending on the year, BoJ URLs sometimes differ slightly. 
        # This targets the main MPM minute index.
        archive_url = f"{base_url}/en/mopo/mpmsche_minu/opinion_{year}/index.htm"
        
        try:
            response = requests.get(archive_url, headers=headers)
            if response.status_code == 404:
                # If a specific year page fails, fall back to the main index
                archive_url = f"{base_url}/en/mopo/mpmsche_minu/index.htm"
                response = requests.get(archive_url, headers=headers)

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all PDF links
            for link in soup.find_all('a', href=True):
                href = link['href']
                # We specifically want the "Summary of Opinions" (opi) in English
                if 'opi' in href and href.endswith('.pdf'):
                    # Handle relative vs absolute URLs
                    if href.startswith('/'):
                        full_url = base_url + href
                    else:
                        full_url = base_url + '/en/mopo/mpmsche_minu/' + href
                        
                    if full_url not in pdf_urls:
                        pdf_urls.append(full_url)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error crawling BoJ year {year}: {e}")

    print(f"Found {len(pdf_urls)} BoJ Summary of Opinions PDFs.")
    return pdf_urls

# 3. Extract Date from BoJ URL
def extract_boj_date(url):
    # BoJ URLs usually look like .../opi231219.pdf (YYMMDD)
    match = re.search(r'opi(\d{2})(\d{2})(\d{2})\.pdf', url)
    if match:
        yy, mm, dd = match.groups()
        # Convert YY to YYYY (assuming 20xx)
        return f"20{yy}-{mm}-{dd}"
    return "Unknown"

# 4. In-Memory PDF Parser
def parse_and_store_pdfs(urls, conn):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    cursor = conn.cursor()
    success_count = 0
    
    for url in urls:
        date_str = extract_boj_date(url)
        print(f"Downloading and parsing PDF for {date_str}...")
        
        try:
            # Step A: Download the raw bytes of the PDF
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Step B: Load bytes directly into PyMuPDF
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            
            full_text = ""
            # Step C: Iterate through every page and extract text
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                full_text += page.get_text("text") + " "
                
            # Clean up the text (remove excessive newlines and spaces from PDF formatting)
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Step D: Save to SQLite
            cursor.execute('''
                INSERT OR IGNORE INTO text_data (date, source, doc_type, text_content)
                VALUES (?, ?, ?, ?)
            ''', (date_str, 'Bank of Japan', 'Summary of Opinions', clean_text))
            
            if cursor.rowcount > 0:
                success_count += 1
                
            pdf_document.close()
            time.sleep(2) # PDFs are heavier, wait 2 seconds between downloads
            
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    conn.commit()
    print(f"\nBoJ Pipeline Complete! Added {success_count} new Japanese policy documents to the database.")

if __name__ == "__main__":
    db_conn = get_db_connection()
    
    # 2016 is when they officially started the "Summary of Opinions" format
    boj_links = get_boj_pdf_links(2016, 2026)
    
    if boj_links:
        parse_and_store_pdfs(boj_links, db_conn)
        
    db_conn.close()