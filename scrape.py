import requests
from bs4 import BeautifulSoup # type: ignore
import os
from pathlib import Path
import time
from urllib.parse import urljoin
import re
from typing import List, Set
import logging

class IrishStatuteBookScraper:
    def __init__(self, download_path: str = "./downloads", delay: float = 1.0):
        self.download_path = Path(download_path)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.download_path.mkdir(parents=True, exist_ok=True)

    def read_links_file(self, file_path: str) -> List[str]:
        urls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and 'http' in line:
                        if ' : ' in line:
                            url = line.split(' : ')[1].strip()
                        elif line.startswith('http'):
                            url = line
                        else:
                            import re
                            url_match = re.search(r'https?://[^\s]+', line)
                            if url_match:
                                url = url_match.group(0)
                            else:
                                continue
                        urls.append(url)
            self.logger.info(f"Found {len(urls)} URLs in {file_path}")
            return urls
        except FileNotFoundError:
            self.logger.error(f"Links file not found: {file_path}")
            return []

    def get_year_from_url(self, url: str) -> str:
        match = re.search(r'/(\d{4})/', url)
        return match.group(1) if match else "unknown"

    def fetch_page(self, url: str) -> BeautifulSoup:
        try:
            self.logger.info(f"Fetching page: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            time.sleep(self.delay)  
            return soup
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def find_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        pdf_links = []
        
        try:
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('/pdf') or 'pdf' in href.lower():
                    full_url = urljoin(base_url, href)
                    pdf_links.append(full_url)
            
            rows = soup.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                for cell in cells:
                    links = cell.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if '/eli/' in href and '/si/' in href:
                            if not href.endswith('/pdf'):
                                pdf_url = href.rstrip('/') + '/made/en/pdf'
                            else:
                                pdf_url = href
                            
                            full_pdf_url = urljoin(base_url, pdf_url)
                            pdf_links.append(full_pdf_url)
            
            base_pattern = re.search(r'(https?://[^/]+/eli/\d{4}/si)', base_url)
            if base_pattern:
                base_eli_url = base_pattern.group(1)
                
                number_links = soup.find_all('a', href=re.compile(r'/eli/\d{4}/si/\d+'))
                for link in number_links:
                    href = link['href']
                    number_match = re.search(r'/si/(\d+)', href)
                    if number_match:
                        number = number_match.group(1)
                        pdf_url = f"{base_eli_url}/{number}/made/en/pdf"
                        pdf_links.append(pdf_url)
            
            unique_links = []
            seen = set()
            for link in pdf_links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)
            
            self.logger.info(f"Found {len(unique_links)} PDF links")
            return unique_links
            
        except Exception as e:
            self.logger.error(f"Error finding PDF links: {e}")
            return []

    def download_pdf(self, pdf_url: str, save_path: Path) -> bool:
        try:
            self.logger.info(f"Downloading: {pdf_url}")
            
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                self.logger.warning(f"URL might not be a PDF: {pdf_url} (content-type: {content_type})")
            
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            file_size = save_path.stat().st_size
            self.logger.info(f"Downloaded: {save_path.name} ({file_size:,} bytes)")
            
            time.sleep(self.delay) 
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Error downloading {pdf_url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {pdf_url}: {e}")
            return False

    def generate_filename(self, year: str, sequence_number: int) -> str:
        return f"{year}_{sequence_number:04d}.pdf"

    def scrape_year(self, url: str) -> int:
        year = self.get_year_from_url(url)
        year_dir = self.download_path / year
        year_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Processing year {year}: {url}")
        
        soup = self.fetch_page(url)
        if not soup:
            return 0
        
        pdf_links = self.find_pdf_links(soup, url)
        
        downloaded_count = 0
        for i, pdf_url in enumerate(pdf_links, 1):
            filename = self.generate_filename(year, i)
            save_path = year_dir / filename
            
            if save_path.exists():
                self.logger.info(f"Skipping existing file: {filename}")
                continue
            
            self.logger.info(f"Progress: {i}/{len(pdf_links)} - {filename}")
            
            if self.download_pdf(pdf_url, save_path):
                downloaded_count += 1
            else:
                alternative_urls = self.get_alternative_pdf_urls(pdf_url)
                for alt_url in alternative_urls:
                    if self.download_pdf(alt_url, save_path):
                        downloaded_count += 1
                        break
        
        self.logger.info(f"Year {year} complete: {downloaded_count}/{len(pdf_links)} files downloaded")
        return downloaded_count

    def get_alternative_pdf_urls(self, original_url: str) -> List[str]:
        alternatives = []
        
        if '/en/pdf' in original_url:
            alternatives.append(original_url.replace('/en/pdf', '/ga/pdf'))
        
        if '/made/en/pdf' in original_url:
            alternatives.append(original_url.replace('/made/en/pdf', '/en/pdf'))
            alternatives.append(original_url.replace('/made/en/pdf', '/pdf'))
        
        return alternatives

    def run(self, links_file: str = "Links.txt") -> None:
        self.logger.info("Starting Irish Statute Book PDF scraper")
        
        urls = self.read_links_file(links_file)
        if not urls:
            self.logger.error("No URLs found. Exiting.")
            return
        
        total_downloaded = 0
        
        for url in urls:
            try:
                downloaded = self.scrape_year(url)
                total_downloaded += downloaded
            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")
                continue
        
        self.logger.info(f"Scraping complete! Total files downloaded: {total_downloaded}")
        self.logger.info(f"Files saved to: {self.download_path.absolute()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Irish Statute Book PDF Scraper")
    parser.add_argument("--download-path", "-d", default="./downloads", 
                      help="Path to save downloaded PDFs (default: ./downloads)")
    parser.add_argument("--delay", "-t", type=float, default=1.0,
                      help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--links-file", "-l", default="Links.txt",
                      help="Path to links file (default: Links.txt)")
    
    args = parser.parse_args()
    
    scraper = IrishStatuteBookScraper(
        download_path=args.download_path,
        delay=args.delay
    )
    
    scraper.run(args.links_file)


if __name__ == "__main__":
    main()