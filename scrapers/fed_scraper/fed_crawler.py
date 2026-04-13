import requests
from bs4 import BeautifulSoup
import time

def get_historical_links(start_year, end_year):
    base_url = "https://www.federalreserve.gov"
    statement_urls = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for year in range(start_year, end_year + 1):
        print(f"Scanning the {year} FOMC Archive...")
        archive_url = f"{base_url}/monetarypolicy/fomchistorical{year}.htm"
        
        try:
            response = requests.get(archive_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Filter ONLY for FOMC Press Release statements
                # The Fed uniquely names these files ending in 'a.htm'
                if 'newsevents/pressreleases/monetary' in href and 'a.htm' in href:
                    full_url = base_url + href
                    
                    # Prevent duplicates
                    if full_url not in statement_urls:
                        statement_urls.append(full_url)
                        
            # Be polite to the Fed's servers
            time.sleep(1) 
            
        except Exception as e:
            print(f"Could not load archive for {year}. Error: {e}")

    print(f"\nTotal Statement Links Found: {len(statement_urls)}")
    return statement_urls

# Run the test for 2010 to 2014
if __name__ == "__main__":
    links = get_historical_links(2010, 2014)
    
    print("\nSample URLs extracted:")
    for l in links[:5]:
        print(l)