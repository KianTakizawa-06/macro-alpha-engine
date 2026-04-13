import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re

# 1. Database Setup
def setup_database():
    conn = sqlite3.connect('macro_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            source TEXT,
            doc_type TEXT,
            text_content TEXT UNIQUE  -- Added UNIQUE to prevent duplicates if you run it twice
        )
    ''')
    conn.commit()
    return conn

# 2. Link Crawler
def get_historical_links(start_year, end_year):
    base_url = "https://www.federalreserve.gov"
    statement_urls = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    print(f"Crawling Fed archives from {start_year} to {end_year}...")
    for year in range(start_year, end_year + 1):
        archive_url = f"{base_url}/monetarypolicy/fomchistorical{year}.htm"
        
        try:
            response = requests.get(archive_url, headers=headers)
            if response.status_code != 200:
                continue # Skip years that don't exist yet or fail
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'newsevents/pressreleases/monetary' in href and 'a.htm' in href:
                    full_url = base_url + href
                    if full_url not in statement_urls:
                        statement_urls.append(full_url)
            time.sleep(1) # Be polite to the server
            
        except Exception as e:
            print(f"Error crawling {year}: {e}")

    print(f"Found {len(statement_urls)} official statements.")
    return statement_urls

# 3. URL Date Extractor
def extract_date_from_url(url):
    # Searches for an 8-digit number in the URL (YYYYMMDD)
    match = re.search(r'(\d{8})', url)
    if match:
        date_raw = match.group(1)
        # Format as YYYY-MM-DD
        return f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    return "Unknown"

# 4. Text Scraper & DB Loader
def scrape_and_store(urls, conn):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    cursor = conn.cursor()
    
    success_count = 0
    
    for url in urls:
        date_str = extract_date_from_url(url)
        print(f"Scraping statement from {date_str}...")
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_div = soup.find('div', id='article')
            
            if article_div:
                raw_text = article_div.get_text(separator=' ', strip=True)
                
                # Insert into database (IGNORE if it already exists due to UNIQUE constraint)
                cursor.execute('''
                    INSERT OR IGNORE INTO text_data (date, source, doc_type, text_content)
                    VALUES (?, ?, ?, ?)
                ''', (date_str, 'Federal Reserve', 'Statement', raw_text))
                
                # If row_count > 0, it was actually inserted (not ignored)
                if cursor.rowcount > 0:
                    success_count += 1
                    
            time.sleep(1) # Crucial: Don't hammer the Fed's servers
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            
    conn.commit()
    print(f"\nPipeline Complete! Successfully added {success_count} new records to the database.")

# 5. Master Execution
if __name__ == "__main__":
    # Initialize DB
    db_conn = setup_database()
    
    # Run the Crawler (Let's do 2010 through 2025)
    links_to_scrape = get_historical_links(2010, 2025)
    
    # Run the Scraper
    if links_to_scrape:
        scrape_and_store(links_to_scrape, db_conn)
        
    db_conn.close()