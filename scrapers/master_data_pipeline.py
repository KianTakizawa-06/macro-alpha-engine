import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import fitz  # PyMuPDF

# ==========================================
# 1. SHARED DATABASE SETUP
# ==========================================
def setup_database():
    conn = sqlite3.connect('macro_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            source TEXT,
            doc_type TEXT,
            text_content TEXT UNIQUE
        )
    ''')
    conn.commit()
    return conn

# ==========================================
# 2. FEDERAL RESERVE PIPELINE (HTML)
# ==========================================
def get_fed_links(start_year, end_year):
    base_url = "https://www.federalreserve.gov"
    urls = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    print(f"\n--- Starting Fed Crawler ({start_year}-{end_year}) ---")
    
    for year in range(start_year, end_year + 1):
        archive_url = f"{base_url}/monetarypolicy/fomchistorical{year}.htm"
        try:
            response = requests.get(archive_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if 'newsevents/pressreleases/monetary' in href and 'a.htm' in href:
                        full_url = base_url + href
                        if full_url not in urls:
                            urls.append(full_url)
            time.sleep(1)
        except Exception as e:
            print(f"Fed Crawl Error {year}: {e}")
    return urls

def scrape_fed_statements(urls, conn):
    headers = {'User-Agent': 'Mozilla/5.0'}
    cursor = conn.cursor()
    success_count = 0
    
    for url in urls:
        match = re.search(r'(\d{8})', url)
        date_str = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}" if match else "Unknown"
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_div = soup.find('div', id='article')
            
            if article_div:
                raw_text = article_div.get_text(separator=' ', strip=True)
                cursor.execute('''
                    INSERT OR IGNORE INTO text_data (date, source, doc_type, text_content)
                    VALUES (?, ?, ?, ?)
                ''', (date_str, 'Federal Reserve', 'Statement', raw_text))
                
                if cursor.rowcount > 0:
                    success_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Fed Scrape Error {date_str}: {e}")
            
    conn.commit()
    print(f"Fed Pipeline complete! {success_count} new statements added.")

# ==========================================
# 3. BANK OF JAPAN PIPELINE (PDF)
# ==========================================
def get_boj_links(start_year, end_year):
    base_url = "https://www.boj.or.jp"
    urls = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    print(f"\n--- Starting BoJ Crawler ({start_year}-{end_year}) ---")
    
    for year in range(start_year, end_year + 1):
        archive_url = f"{base_url}/en/mopo/mpmsche_minu/opinion_{year}/index.htm"
        try:
            response = requests.get(archive_url, headers=headers)
            if response.status_code == 404:
                archive_url = f"{base_url}/en/mopo/mpmsche_minu/index.htm"
                response = requests.get(archive_url, headers=headers)

            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'opi' in href and href.endswith('.pdf'):
                    full_url = base_url + href if href.startswith('/') else base_url + '/en/mopo/mpmsche_minu/' + href
                    if full_url not in urls:
                        urls.append(full_url)
            time.sleep(1)
        except Exception as e:
            print(f"BoJ Crawl Error {year}: {e}")
    return urls

def scrape_boj_pdfs(urls, conn):
    headers = {'User-Agent': 'Mozilla/5.0'}
    cursor = conn.cursor()
    success_count = 0
    
    for url in urls:
        match = re.search(r'opi(\d{2})(\d{2})(\d{2})\.pdf', url)
        date_str = f"20{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else "Unknown"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            full_text = " ".join([page.get_text("text") for page in pdf_document])
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            
            cursor.execute('''
                INSERT OR IGNORE INTO text_data (date, source, doc_type, text_content)
                VALUES (?, ?, ?, ?)
            ''', (date_str, 'Bank of Japan', 'Summary of Opinions', clean_text))
            
            if cursor.rowcount > 0:
                success_count += 1
                
            pdf_document.close()
            time.sleep(2)
        except Exception as e:
            print(f"BoJ PDF Error {date_str}: {e}")

    conn.commit()
    print(f"BoJ Pipeline complete! {success_count} new summaries added.")

# ==========================================
# 4. MISSION CONTROL
# ==========================================
if __name__ == "__main__":
    print("Initializing Macro Engine Data Pipeline...")
    db_conn = setup_database()
    
    # 1. Run Federal Reserve Pipeline (2010 to Present)
    fed_urls = get_fed_links(2010, 2026)
    if fed_urls:
        scrape_fed_statements(fed_urls, db_conn)
        
    # 2. Run Bank of Japan Pipeline (2016 to Present)
    boj_urls = get_boj_links(2016, 2026)
    if boj_urls:
        scrape_boj_pdfs(boj_urls, db_conn)
        
    db_conn.close()
    print("\nAll data successfully securely loaded into macro_engine.db!")