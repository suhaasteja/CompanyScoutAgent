import os
import pandas as pd
import re
from docling.document_converter import DocumentConverter

csv_path = "domain_urls.csv"
output_path = "website_scrape.csv"

def scrape(url):
    converter = DocumentConverter()
    result = converter.convert(url)
    scraped_markdown = result.document.export_to_markdown()
    
    # Remove Unicode characters (non-ASCII)
    scraped_markdown = scraped_markdown.encode("ascii", "ignore").decode()
    
    # Remove unwanted placeholders
    scraped_markdown = scraped_markdown.replace("<!-- image -->", "")
    
    return scraped_markdown

df = pd.read_csv(csv_path)
df_new = pd.DataFrame(columns=["url", "title", "markdown"])

for index, row in df.iterrows():
    url = row['url']
    title = row['title']
    scraped_markdown = scrape(url)

    print(f"Scraped {url} \n\n {scraped_markdown[:500]}... \n\n")  # Preview first 500 chars

    # Append row properly
    df_new.loc[index] = [url, title, scraped_markdown]

df_new.to_csv(output_path, index=False)
