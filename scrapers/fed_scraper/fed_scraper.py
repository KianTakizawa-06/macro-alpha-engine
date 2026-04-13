import requests
from bs4 import BeautifulSoup
import sqlite3

# 1. Initialize the SQLite Database
def setup_database():
    # This creates a file called 'macro_engine.db' in your folder
    conn = sqlite3.connect('macro_engine.db')
    cursor = conn.cursor()
    
    # Create the table for our text data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            source TEXT,
            doc_type TEXT,
            text_content TEXT
        )
    ''')
    conn.commit()
    return conn

# 2. The Scraping Engine
def scrape_fomc_statement(url, date_str, conn):
    print(f"Scraping FOMC Statement for {date_str}...")
    
    # We use a header to pretend we are a standard web browser
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check if the page actually loaded
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # The Fed keeps their main text inside a div with the id "article"
        article_div = soup.find('div', id='article')
        
        if article_div:
            # Extract the text and strip out the HTML tags
            raw_text = article_div.get_text(separator=' ', strip=True)
            
            # Save it to our database
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO text_data (date, source, doc_type, text_content)
                VALUES (?, ?, ?, ?)
            ''', (date_str, 'Federal Reserve', 'Statement', raw_text))
            conn.commit()
            
            print(f"Success! Saved {len(raw_text)} characters to the database.")
        else:
            print("Error: Could not find the article text on the page.")
            
    except Exception as e:
        print(f"Failed to scrape the page. Error: {e}")

# 3. Run the Pipeline
if __name__ == "__main__":
    # Setup DB
    db_connection = setup_database()
    
    # Target URL: The pivotal July 2023 rate hike statement
    target_url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230726a.htm"
    statement_date = "2023-07-26"
    
    # Execute
    scrape_fomc_statement(target_url, statement_date, db_connection)
    
    # Close the database safely
    db_connection.close()