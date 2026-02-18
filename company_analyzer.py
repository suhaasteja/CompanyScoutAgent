"""
Company Analyzer - Extract structured data from company websites with source citations
"""
import os
import json
import re
from typing import Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Client will be passed in or created from env
_client = None

def get_client(api_key: str = None):
    """Get or create OpenAI client."""
    global _client
    if api_key:
        return OpenAI(api_key=api_key)
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class CitedValue:
    """A value with its source URL."""
    value: str
    source: Optional[str] = None

    def __str__(self):
        return self.value


@dataclass
class CitedList:
    """A list of values with sources."""
    items: list[dict] = field(default_factory=list)  # [{value: str, source: str}, ...]

    def values(self) -> list[str]:
        return [item.get("value", "") for item in self.items]

    def __iter__(self):
        return iter(self.items)

    def __bool__(self):
        return len(self.items) > 0


@dataclass
class CompanyProfile:
    name: str
    description: CitedValue
    founding_year: CitedValue
    founders: CitedList
    headquarters: CitedValue
    industry: CitedValue
    business_model: CitedValue
    tech_stack: CitedList
    key_products: CitedList
    target_customers: CitedValue
    funding_stage: CitedValue
    notable_investors: CitedList
    team_size: CitedValue
    hiring_signals: CitedList
    competitive_advantages: CitedList
    potential_risks: CitedList
    source_urls: list[str] = field(default_factory=list)


EXTRACTION_PROMPT = """You are a VC analyst assistant. Extract structured company information from the website content below.

IMPORTANT: For each piece of information, include the source URL where you found it.
The content is organized with URLs marked as "=== URL ===" headers.

Be precise and only include information explicitly stated in the content.
If information is not available, use null.

Extract the following (each with "value" and "source" URL):

- name: Company name
- description: {{value: "1-2 sentence description", source: "URL"}}
- founding_year: {{value: "year", source: "URL"}}
- founders: [{{value: "founder name", source: "URL"}}, ...]
- headquarters: {{value: "location", source: "URL"}}
- industry: {{value: "industry", source: "URL"}}
- business_model: {{value: "how they make money", source: "URL"}}
- tech_stack: [{{value: "technology", source: "URL"}}, ...]
- key_products: [{{value: "product name", source: "URL"}}, ...]
- target_customers: {{value: "customer segment", source: "URL"}}
- funding_stage: {{value: "stage", source: "URL"}}
- notable_investors: [{{value: "investor name", source: "URL"}}, ...]
- team_size: {{value: "size", source: "URL"}}
- hiring_signals: [{{value: "signal", source: "URL"}}, ...]
- competitive_advantages: [{{value: "advantage", source: "URL"}}, ...]
- potential_risks: [{{value: "risk", source: "URL"}}, ...]

Website Content:
{content}

Available source URLs from the scrape:
{urls}

Respond with valid JSON only. Use actual URLs from the list above as sources."""


def extract_urls_from_content(content: str) -> list[str]:
    """Extract all URLs from the scraped content markers."""
    pattern = r'=== (https?://[^\s]+) ==='
    return list(set(re.findall(pattern, content)))


def analyze_company(content: str, url: str, api_key: str = None) -> CompanyProfile:
    """Extract structured company data with source citations."""
    client = get_client(api_key)

    # Get all source URLs
    source_urls = extract_urls_from_content(content)
    if not source_urls:
        source_urls = [url]

    # Truncate content if too long
    max_chars = 40000
    truncated_content = content[:max_chars] if len(content) > max_chars else content

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a precise data extraction assistant. Always respond with valid JSON. Always include source URLs for each piece of information."
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    content=truncated_content,
                    urls="\n".join(source_urls)
                )
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        data = {"name": "Unknown", "description": {"value": "Could not parse", "source": url}}

    # Extract name as plain string (LLM sometimes returns it as dict)
    raw_name = data.get("name", "Unknown")
    if isinstance(raw_name, dict):
        company_name = raw_name.get("value", "Unknown")
    else:
        company_name = str(raw_name) if raw_name else "Unknown"

    # Helper to parse cited value
    def parse_cited(field_data, default="") -> CitedValue:
        if isinstance(field_data, dict):
            return CitedValue(
                value=str(field_data.get("value", default) or default),
                source=field_data.get("source")
            )
        elif isinstance(field_data, str):
            return CitedValue(value=field_data, source=None)
        return CitedValue(value=default, source=None)

    # Helper to parse cited list
    def parse_cited_list(field_data) -> CitedList:
        if isinstance(field_data, list):
            items = []
            for item in field_data:
                if isinstance(item, dict):
                    items.append({
                        "value": str(item.get("value", "")),
                        "source": item.get("source")
                    })
                elif isinstance(item, str):
                    items.append({"value": item, "source": None})
            return CitedList(items=items)
        return CitedList(items=[])

    profile = CompanyProfile(
        name=company_name,
        description=parse_cited(data.get("description"), ""),
        founding_year=parse_cited(data.get("founding_year")),
        founders=parse_cited_list(data.get("founders", [])),
        headquarters=parse_cited(data.get("headquarters")),
        industry=parse_cited(data.get("industry"), "Unknown"),
        business_model=parse_cited(data.get("business_model"), "Unknown"),
        tech_stack=parse_cited_list(data.get("tech_stack", [])),
        key_products=parse_cited_list(data.get("key_products", [])),
        target_customers=parse_cited(data.get("target_customers"), "Unknown"),
        funding_stage=parse_cited(data.get("funding_stage")),
        notable_investors=parse_cited_list(data.get("notable_investors", [])),
        team_size=parse_cited(data.get("team_size")),
        hiring_signals=parse_cited_list(data.get("hiring_signals", [])),
        competitive_advantages=parse_cited_list(data.get("competitive_advantages", [])),
        potential_risks=parse_cited_list(data.get("potential_risks", [])),
        source_urls=source_urls
    )

    return profile


def format_cited_value(cv: CitedValue, label: str = None) -> str:
    """Format a cited value as markdown with optional link."""
    if not cv.value:
        return "N/A"
    if cv.source:
        return f"{cv.value} [[source]({cv.source})]"
    return cv.value


def format_cited_list_md(cl: CitedList) -> str:
    """Format a cited list as markdown bullets with sources."""
    if not cl.items:
        return "Not found"
    lines = []
    for item in cl.items:
        val = item.get("value", "")
        src = item.get("source")
        if src:
            lines.append(f"- {val} [[source]({src})]")
        else:
            lines.append(f"- {val}")
    return "\n".join(lines)


def profile_to_markdown(profile: CompanyProfile) -> str:
    """Convert a CompanyProfile to formatted markdown with citations."""

    founders_md = format_cited_list_md(profile.founders) if profile.founders else "Not found"
    products_md = format_cited_list_md(profile.key_products) if profile.key_products else "Not found"
    advantages_md = format_cited_list_md(profile.competitive_advantages) if profile.competitive_advantages else "Not identified"
    risks_md = format_cited_list_md(profile.potential_risks) if profile.potential_risks else "None identified"
    hiring_md = format_cited_list_md(profile.hiring_signals) if profile.hiring_signals else "No signals detected"

    tech_items = []
    for item in profile.tech_stack.items if profile.tech_stack else []:
        val = item.get("value", "")
        src = item.get("source")
        if src:
            tech_items.append(f"[{val}]({src})")
        else:
            tech_items.append(val)
    tech_md = ", ".join(tech_items) if tech_items else "Not found"

    investors_items = []
    for item in profile.notable_investors.items if profile.notable_investors else []:
        val = item.get("value", "")
        src = item.get("source")
        if src:
            investors_items.append(f"[{val}]({src})")
        else:
            investors_items.append(val)
    investors_md = ", ".join(investors_items) if investors_items else "Not found"

    md = f"""## {profile.name}

**{format_cited_value(profile.description)}**

---

### Overview
| Field | Value |
|-------|-------|
| Industry | {format_cited_value(profile.industry)} |
| Business Model | {format_cited_value(profile.business_model)} |
| Headquarters | {format_cited_value(profile.headquarters) if profile.headquarters.value else 'N/A'} |
| Founded | {format_cited_value(profile.founding_year) if profile.founding_year.value else 'N/A'} |
| Team Size | {format_cited_value(profile.team_size) if profile.team_size.value else 'N/A'} |
| Funding Stage | {format_cited_value(profile.funding_stage) if profile.funding_stage.value else 'N/A'} |

### Founders
{founders_md}

### Key Products
{products_md}

### Target Customers
{format_cited_value(profile.target_customers)}

### Tech Stack
{tech_md}

### Investors
{investors_md}

### Hiring Signals
{hiring_md}

### Competitive Advantages
{advantages_md}

### Potential Risks
{risks_md}

---
**Sources:** {', '.join(f'[{i+1}]({url})' for i, url in enumerate(profile.source_urls[:5]))}
"""
    return md


def profile_to_dict(profile: CompanyProfile) -> dict:
    """Convert profile to dictionary for JSON serialization."""
    return {
        "name": profile.name,
        "description": {"value": profile.description.value, "source": profile.description.source},
        "founding_year": {"value": profile.founding_year.value, "source": profile.founding_year.source},
        "founders": profile.founders.items,
        "headquarters": {"value": profile.headquarters.value, "source": profile.headquarters.source},
        "industry": {"value": profile.industry.value, "source": profile.industry.source},
        "business_model": {"value": profile.business_model.value, "source": profile.business_model.source},
        "tech_stack": profile.tech_stack.items,
        "key_products": profile.key_products.items,
        "target_customers": {"value": profile.target_customers.value, "source": profile.target_customers.source},
        "funding_stage": {"value": profile.funding_stage.value, "source": profile.funding_stage.source},
        "notable_investors": profile.notable_investors.items,
        "team_size": {"value": profile.team_size.value, "source": profile.team_size.source},
        "hiring_signals": profile.hiring_signals.items,
        "competitive_advantages": profile.competitive_advantages.items,
        "potential_risks": profile.potential_risks.items,
        "source_urls": profile.source_urls
    }
