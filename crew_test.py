import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import GoogleSerperAPIWrapper
from docling.document_converter import DocumentConverter
from scraper import scrape

# Set up your SERPER_API_KEY key in an .env file
load_dotenv()

# Function definitions
def create_search_tool():
    search = GoogleSerperAPIWrapper()
    
    class SearchTool(BaseTool):
        name: str = "Search"
        description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
        search: GoogleSerperAPIWrapper = Field(default_factory=lambda: search)

        def _run(self, query: str) -> str:
            """Execute the search query and return results"""
            try:
                search_results = self.search.results(query)
                
                print("before" ,search_results)
                
                # Extract the organic search results
                organic_results = search_results.get("organic", [])
                
                # Get the top 5 URLs
                top_urls = []
                for result in organic_results[:3]:
                    if "link" in result:
                        top_urls.append(result["link"])
                
                print("after", top_urls)
                
                return "\n".join(top_urls) if top_urls else "No results found."
                
            except Exception as e:
                return f"Error performing search: {str(e)}"
    
    return SearchTool()

def create_docling_scrape_tool():
    class DoclingScrapeTool(BaseTool):
        name: str = "DoclingWebScrape"
        description: str = "Scrapes content from a URL using Docling to get more detailed information. Use after finding relevant URLs through Search."
        
        def _run(self, url: str) -> str:
            """Scrape content from the provided URL using Docling"""
            try:
                scraped_content = scrape(url)
                return f"Content from {url}:\n{scraped_content}"
                
            except Exception as e:
                return f"Error scraping URL {url}: {str(e)}"
    
    return DoclingScrapeTool()

def create_researcher_agent():
    # Create the tools
    search_tool = create_search_tool()
    docling_tool = create_docling_scrape_tool()
    
    # Create the agent
    researcher = Agent(
        role='Research Analyst',
        goal='Gather current market data and trends',
        backstory="""You are an expert research analyst with years of experience in
        gathering market intelligence. Conduct comprehensive market analysis on FAANGs i.e Facebook (now Meta), Amazon, Apple, Netflix, Google.
        Gather data on market size, growth projections, key players, and emerging trends.
        Generate a report describing which one is a better investment as of Q1 2025. You're known for your ability to find
        relevant and up-to-date market information and present it in a clear,
        actionable format.""",
        tools=[search_tool, docling_tool],
        verbose=True
    )
    
    return researcher

def create_research_task(researcher):
    return Task(
        description="""Conduct comprehensive market analysis on FAANGs.
        Gather data on market size, growth projections, key players, and emerging trends.
        Generate a report describing which one is a better investment as of Q1 2025.""",
        agent=researcher,
        expected_output="A detailed market research report with current data and insights."
    )

def run_faang_research():
    researcher = create_researcher_agent()
    research_task = create_research_task(researcher)
    
    # Create a Crew
    research_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    # Execute the crew's work
    result = research_crew.kickoff()
    return result

# This block only runs when the script is executed directly
if __name__ == "__main__":
    researcher = create_researcher_agent()
    print(f"LLM using {researcher.llm.model}")
    
    result = run_faang_research()
    print(result)