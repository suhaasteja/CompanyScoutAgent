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
    
    research_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    result = research_crew.kickoff()
    return result

def create_linkedin_content_agent():
    search_tool = create_search_tool()
    docling_tool = create_docling_scrape_tool()
    
    linkedin_agent = Agent(
        role='LinkedIn Content Creator',
        goal='Create engaging LinkedIn content about AI for mid-market companies',
        backstory="""You are an expert in both AI technology research and content creation.
        You understand how artificial intelligence is transforming mid-market businesses and
        can translate technical concepts into engaging social media content. You're skilled at
        researching the latest trends, finding case studies, and crafting compelling LinkedIn
        posts that drive engagement with business professionals. You know how to optimize content
        for SEO and include relevant data points to build credibility.""",
        tools=[search_tool, docling_tool],
        verbose=True
    )
    
    return linkedin_agent

def create_linkedin_content_task(agent):
    return Task(
        description="""Research AI trends for mid-market companies and create LinkedIn content options.
        
        Step 1: Research the latest AI trends specifically relevant to mid-market companies.
        Focus on practical applications, ROI statistics, case studies, and competitive advantages.
        
        Step 2: Create 3 different LinkedIn post options based on your research.
        Each post should focus on a different AI trend relevant to mid-market companies.
        
        For each post:
        1. Create a compelling headline that grabs attention
        2. Write an engaging opening hook
        3. Present key insights and data points
        4. Include a clear call-to-action
        5. Add 3-5 relevant hashtags
        
        Keep each post within LinkedIn's optimal length (approximately 1,300 characters).
        Optimize the content for SEO and engagement.""",
        agent=agent,
        expected_output="Three distinct LinkedIn post options, each focusing on a different AI trend for mid-market companies."
    )

def run_linkedin_content_generation():
    linkedin_agent = create_linkedin_content_agent()
    linkedin_task = create_linkedin_content_task(linkedin_agent)
    
    linkedin_crew = Crew(
        agents=[linkedin_agent],
        tasks=[linkedin_task],
        verbose=True
    )
    
    result = linkedin_crew.kickoff()
    return result

if __name__ == "__main__":
    researcher = create_researcher_agent()
    print(f"LLM using {researcher.llm.model}")
    
    # result = run_faang_research()
    # print(result)