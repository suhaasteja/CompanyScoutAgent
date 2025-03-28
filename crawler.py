import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
import csv
import os
import argparse


class DomainCrawlerSpider(scrapy.Spider):
    name = "domain_crawler"
    
    def __init__(self, start_url=None, max_depth=2, output_file='domain_urls.csv', delay=1, 
                 respect_robots=True, user_agent=None):
        super(DomainCrawlerSpider, self).__init__()
        
        if not start_url:
            raise ValueError("You must provide a start_url parameter")
        
        # Set the start URL
        self.start_urls = [start_url]
        
        # Parse the domain from the start URL
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        self.base_domain = parsed_url.netloc
        
        # Set max depth
        self.max_depth = int(max_depth)
        
        # Set output file
        self.csv_file = output_file
        
        # Set crawler settings
        self.custom_settings = {
            'DOWNLOAD_DELAY': delay,
            'ROBOTSTXT_OBEY': respect_robots,
            'COOKIES_ENABLED': False,
            'LOG_LEVEL': 'INFO'
        }
        
        if user_agent:
            self.custom_settings['USER_AGENT'] = user_agent
        
        self.visited_urls = set()
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['url', 'depth', 'title', 'status'])
    
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, meta={'depth': 0})
    
    def parse(self, response):
        current_depth = response.meta.get('depth', 0)
        
        current_url = response.url
        
        if current_url in self.visited_urls:
            return

        self.visited_urls.add(current_url)
        
        title = response.css('title::text').get() or "No Title"
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([current_url, current_depth, title, response.status])
        
        if self.max_depth != -1 and current_depth >= self.max_depth:
            return
        
        links = response.css('a::attr(href)').getall()
        
        for link in links:
            absolute_url = response.urljoin(link)
            parsed_url = urlparse(absolute_url)
            
            if parsed_url.netloc == self.base_domain or parsed_url.netloc.endswith('.' + self.base_domain):
                if absolute_url not in self.visited_urls:
                    yield scrapy.Request(
                        absolute_url,
                        callback=self.parse,
                        meta={'depth': current_depth + 1}
                    )


def run_crawler(start_url, max_depth=2, output_file='domain_urls.csv', delay=1, 
               respect_robots=True, user_agent=None):
    process = CrawlerProcess()
    
    process.crawl(
        DomainCrawlerSpider, 
        start_url=start_url, 
        max_depth=max_depth,
        output_file=output_file,
        delay=delay,
        respect_robots=respect_robots,
        user_agent=user_agent
    )
    
    process.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl a domain and its subdomains.')
    parser.add_argument('start_url', help='The URL to start crawling from')
    parser.add_argument('--max-depth', type=int, default=2, 
                       help='Maximum crawl depth (default: 2, use -1 for unlimited)')
    parser.add_argument('--output-file', default='domain_urls.csv',
                       help='CSV file to store results (default: domain_urls.csv)')
    parser.add_argument('--delay', type=float, default=1,
                       help='Delay between requests in seconds (default: 1)')
    parser.add_argument('--ignore-robots', action='store_true',
                       help='Ignore robots.txt')
    parser.add_argument('--user-agent', 
                       default='Mozilla/5.0 (compatible; DomainCrawler/1.0)',
                       help='Custom user agent string')
    
    args = parser.parse_args()
    
    run_crawler(
        start_url=args.start_url,
        max_depth=args.max_depth,
        output_file=args.output_file,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
        user_agent=args.user_agent
    )