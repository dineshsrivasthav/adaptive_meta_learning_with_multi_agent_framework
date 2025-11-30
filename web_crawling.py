import os
import json
import time
import re
import schedule
from typing import List, Dict, Set, Any, Optional
from datetime import datetime
from firecrawl import Firecrawl 
from collections import defaultdict
from langchain.schema import Document
from RAG_workflow import get_or_create_vector_db, add_documents_to_vector_db



def get_deepfake_sources() -> Dict[str, List[str]]:
    """A comprehensive collection of deepfake information sources."""
    return {
        'dataset_repositories': [
            'https://paperswithcode.com/datasets?q=deepfake',
            'https://paperswithcode.com/task/deepfake-detection',
            'https://huggingface.co/datasets?search=deepfake',
            'https://www.kaggle.com/search?q=deepfake',
            'https://github.com/topics/deepfake',
            'https://github.com/topics/deepfake-detection',
            'https://archive.org/search.php?query=deepfake',
            'https://zenodo.org/search?q=deepfake',
            'https://datafinder.ai/datasets?q=deepfake'
        ],
        'research_portals': [
            'https://arxiv.org/search/?query=deepfake&searchtype=all',
            'https://arxiv.org/search/?query=synthetic+media&searchtype=all',
            'https://scholar.google.com/scholar?q=deepfake+detection',
            'https://www.semanticscholar.org/search?q=deepfake',
            'https://dl.acm.org/action/doSearch?AllField=deepfake',
            'https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=deepfake',
            'https://pubmed.ncbi.nlm.nih.gov/?term=deepfake'
        ],
        'news_outlets': [
            'https://www.wired.com/tag/deepfakes/',
            'https://www.theverge.com/search?q=deepfake',
            'https://techcrunch.com/tag/deepfakes/',
            'https://arstechnica.com/search/?query=deepfake',
            'https://www.technologyreview.com/topic/artificial-intelligence/',
            'https://www.bbc.com/news/topics/cd1xp2grgjxt',
            'https://www.theguardian.com/technology/artificialintelligenceai',
            'https://www.reuters.com/technology/',
            'https://www.cnet.com/news/tag/deepfake/',
            'https://www.zdnet.com/topic/artificial-intelligence/'
        ],
        'security_blogs': [
            'https://thehackernews.com/?s=deepfake',
            'https://www.bleepingcomputer.com/search/?q=deepfake',
            'https://www.darkreading.com/search?q=deepfake',
            'https://www.securityweek.com/?s=deepfake',
            'https://krebsonsecurity.com/?s=deepfake',
            'https://threatpost.com/?s=deepfake',
            'https://www.schneier.com/?s=deepfake',
            'https://www.csoonline.com/search/?q=deepfake'
        ],
        'ai_research_labs': [
            'https://openai.com/blog/',
            'https://deepmind.google/discover/blog/',
            'https://ai.meta.com/blog/',
            'https://www.anthropic.com/research',
            'https://www.microsoft.com/en-us/research/research-area/artificial-intelligence/',
            'https://research.google/pubs/',
            'https://bair.berkeley.edu/blog/',
            'https://ai.stanford.edu/blog/'
        ],
        'government_resources': [
            'https://www.cisa.gov/search?search_query=deepfake',
            'https://www.fbi.gov/search?q=deepfake',
            'https://www.nist.gov/search?k=deepfake',
            'https://www.dhs.gov/search?query=deepfake',
            'https://www.europol.europa.eu/search?keys=deepfake'
        ],
        'detection_tools': [
            'https://sensity.ai/blog/',
            'https://www.truepic.com/blog',
            'https://reality-defender.com/blog/',
            'https://www.deeptrace.com/',
            'https://github.com/dessa-oss/DeepFake-Detection',
            'https://github.com/deepfakes/faceswap'
        ],
        'academic_institutions': [
            'https://cyber.harvard.edu/topics/misinformation',
            'https://hai.stanford.edu/',
            'https://www.media.mit.edu/',
            'https://nyuad.nyu.edu/en/research/centers-labs-and-projects/center-for-artificial-intelligence-and-robotics.html'
        ],
        'industry_reports': [
            'https://www.gartner.com/en/documents?q=deepfake',
            'https://www.forrester.com/search?q=deepfake',
            'https://www.mckinsey.com/search?q=deepfake',
            'https://www2.deloitte.com/us/en/pages/technology/topics/deepfakes.html'
        ],
        'forums_communities': [
            'https://news.ycombinator.com/from?site=deepfake',
            'https://www.reddit.com/r/deepfakes/',
            'https://www.reddit.com/r/MediaSynthesis/',
            'https://stackoverflow.com/questions/tagged/deepfake'
        ]
    }

# --- Utility Functions ---

def extract_relevant_urls(content_text: str, base_url: str) -> List[str]:
    """Extract and filter potential deepfake-related URLs from text."""
    # Find all HTTP/HTTPS links
    urls = set(re.findall(r'https?://[^\s<>"]+', content_text))
    
    # Filter for relevance to deepfakes/AI research/datasets
    keywords = ['deepfake', 'synthetic', 'detection', 'deepfakes', 'synthetic media', 'AI', 'fake', 'deep-fake','fakes', 'deep fake', 'deep fakes','misinformation']
    
    relevant_urls = []
    for url in urls:
        if url.startswith(base_url):
            # Skip self-links to avoid infinite loops during recursive scraping
            continue 
        if any(k in url.lower() for k in keywords):
            relevant_urls.append(url)
            
    return relevant_urls


# --- Core Scanner Class ---

class DFScanner:
    """Deepfake Scanner using the Firecrawl library."""

    def __init__(self, api_key: str, vector_db_path: str = "./chroma_db", 
                 add_to_vector_db: bool = True):
        if not api_key:
            raise ValueError("Firecrawl API key is required")
            
        # Initialize the Firecrawl client
        self.firecrawl = Firecrawl(api_key=api_key)
        
        self.crawled: Set[str] = set()
        self.all_content: List[Dict[str, Any]] = []
        self.discovered_urls: Set[str] = set()
        self.results = defaultdict(int)
        
        # Vector DB integration
        self.add_to_vector_db = add_to_vector_db
        self.vector_db_path = vector_db_path
        self.db = None
        self.embeddings = None
        
        if self.add_to_vector_db:
            try:
                self.db, self.embeddings = get_or_create_vector_db(
                    persist_directory=self.vector_db_path,
                    pdf_folder=None  # Don't load PDFs here
                )
                print("Vector DB initialized for web crawling content.")
            except Exception as e:
                print(f"Warning: Could not initialize vector DB: {e}")
                self.add_to_vector_db = False

    def _process_pages(self, pages: List[Dict], category: str):
        """Processes a list of pages (from scrape or crawl) for content and links."""
        
        for page in pages:
            url = page.get('url')
            if not url or not page.get('markdown'):
                continue

            # Store the main content
            self.all_content.append({
                'url': url,
                'source_category': category,
                'title': page.get('title', 'No Title'),
                'markdown': page['markdown'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Extract URLs from this page's content
            new_links = extract_relevant_urls(page['markdown'], url)
            self.discovered_urls.update(new_links)
            self.results['links_discovered'] += len(new_links)

    def run_scan(self, deep_crawl: bool = False):
        """Systematically crawl all deepfake information sources in two phases."""
        
        # --- PHASE 1: Crawl Primary Sources ---
        sources = get_deepfake_sources()
        total_sources = sum(len(urls) for urls in sources.values())
        print(f"Starting Phase 1: Processing {total_sources} primary sources...")

        for category, urls in sources.items():
            print(f"Processing category: {category}")
            for url in urls:
                if url in self.crawled:
                    continue
                self.crawled.add(url)
                self.results['sources_processed'] += 1

                is_deep_crawl = deep_crawl and category in ['dataset_repositories', 'research_portals']
                
                try:
                    if is_deep_crawl:
                        pages = self.firecrawl.crawl(
                            url=url, 
                            crawler_options={'max_depth': 2, 'limit': 20},
                            page_options={'onlyMainContent': True, 'includeHTML': False}
                        )
                        self.results['pages_crawled'] += len(pages)
                    else:
                        page = self.firecrawl.scrape(url=url, formats=['markdown'])
                        pages = [page] if page else []
                        self.results['pages_scraped'] += len(pages)
                
                    self._process_pages(pages, category)
                    
                except Exception as e:
                    print(f"Failed to process {url} ({category}): {e.__class__.__name__}")
                    
                time.sleep(2) 

        # --- PHASE 2: Scrape Discovered Relevant URLs ---
        
        # Filter discovered URLs to those not yet crawled
        priority_urls = [url for url in self.discovered_urls if url not in self.crawled]
        
        print(f"\nStarting Phase 2: Scraping {len(priority_urls)} discovered links...")
        
        for url in priority_urls:
            if len(self.all_content) > 1000: 
                print("Content limit reached.")
                break

            if url in self.crawled: continue
            self.crawled.add(url)
            self.results['discovered_urls_scraped'] += 1
            
            try:
                page = self.firecrawl.scrape(url=url, formats=['markdown'])
                if page:
                    self._process_pages([page], 'discovered_content')
            except Exception as e:
                pass 
                
            time.sleep(2) 

    def add_content_to_vector_db(self):
        """Add crawled content to the vector database."""
        if not self.add_to_vector_db or not self.db:
            return
        
        if not self.all_content:
            print("No content to add to vector DB.")
            return
        
        try:
            # Convert crawled content to Document objects
            documents = []
            for content_item in self.all_content:
                doc = Document(
                    page_content=content_item.get('markdown', ''),
                    metadata={
                        'url': content_item.get('url', ''),
                        'source_category': content_item.get('source_category', 'web_crawled'),
                        'title': content_item.get('title', 'No Title'),
                        'timestamp': content_item.get('timestamp', datetime.now().isoformat())
                    }
                )
                documents.append(doc)
            
            # Add documents to vector DB
            add_documents_to_vector_db(self.db, documents, self.embeddings)
            print(f"Successfully added {len(documents)} documents to vector DB.")
            
        except Exception as e:
            print(f"Error adding content to vector DB: {e}")

    def report(self, filename: str = 'deepfake_firecrawl_final_report.json'):
        """Generate and save the final report JSON containing all scraped content."""
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': dict(self.results),
            'total_urls_scraped': len(self.crawled),
            'total_content_stored': len(self.all_content),
            'scraped_content': self.all_content 
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\nScan finished.")
        print(f"Total content pages collected: {len(self.all_content)}")
        print(f"Results saved to: {filename}")
        
        if self.add_to_vector_db:
            self.add_content_to_vector_db()


def run_periodic_scan(scanner: DFScanner, deep_crawl: bool = False, 
                     interval_hours: int = 24):
    """
    Run periodic web scans and update vector DB.
    
    Args:
        scanner: DFScanner instance
        deep_crawl: Whether to use deep crawling
        interval_hours: Hours between scans
    """
    def job():
        print(f"\n{'='*60}")
        print(f"Starting periodic scan at {datetime.now()}")
        print(f"{'='*60}\n")
        try:
            scanner.run_scan(deep_crawl=deep_crawl)
            scanner.report()
        except Exception as e:
            print(f"Error in periodic scan: {e}")
    
    # Schedule the job
    schedule.every(interval_hours).hours.do(job)
    
    # Run initial scan
    job()
    
    # Keep running
    print(f"\nScheduled periodic scans every {interval_hours} hours.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nStopping periodic scans.")


def main():
    """Main execution entry point."""
    print("Deepfake Intelligence Gathering Scan")
    
    key = os.environ.get('FIRECRAWL_API_KEY')
    if not key:
        print("ERROR: FIRECRAWL_API_KEY environment variable not set.")
        return

    vector_db_path = os.environ.get('VECTOR_DB_PATH', './chroma_db')
    add_to_db = os.environ.get('ADD_TO_VECTOR_DB', 'true').lower() == 'true'
    periodic = os.environ.get('PERIODIC_SCAN', 'false').lower() == 'true'
    interval_hours = int(os.environ.get('SCAN_INTERVAL_HOURS', '24'))

    scanner = DFScanner(api_key=key, vector_db_path=vector_db_path, 
                       add_to_vector_db=add_to_db)
    
    if periodic:
        deep_crawl_input = os.environ.get('DEEP_CRAWL', 'n').lower() == 'y'
        run_periodic_scan(scanner, deep_crawl=deep_crawl_input, 
                         interval_hours=interval_hours)
    else:
        deep_crawl_input = input("Use Firecrawl crawl endpoint for deep crawling of research/dataset sites? (y/N): ")
        deep_crawl = deep_crawl_input.lower() == 'y'
        
        try:
            scanner.run_scan(deep_crawl=deep_crawl)
            scanner.report()
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")
            scanner.report('deepfake_firecrawl_interrupted.json')

if __name__ == "__main__":
    main()