"""
CompanyScoutAgent - VC-focused company research tool with caching & citations
"""
import os
import json
import streamlit as st
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

from company_analyzer import (
    analyze_company, profile_to_dict, CompanyProfile,
    CitedValue, CitedList
)
from storage import (
    extract_domain, company_exists, get_company, save_company,
    get_recent_companies, search_company_content, get_data_age,
    delete_company, get_retrieval_stats
)

load_dotenv()

# Initialize OpenAI client (will be set in sidebar if not in env)
def get_openai_client():
    """Get OpenAI client, checking session state first, then env."""
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

client = None  # Will be initialized after checking settings

# Page config
st.set_page_config(
    page_title="CompanyScout",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .company-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .company-header h1 { margin: 0 0 0.5rem 0; font-size: 2rem; }
    .company-header p { margin: 0; opacity: 0.9; font-size: 1.1rem; }

    .stats-row { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
    .stat-chip {
        background: rgba(255,255,255,0.15);
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94a3b8 !important;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }

    .score-card {
        text-align: center;
        padding: 0.8rem;
        background: #f8fafc;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        position: relative;
        cursor: help;
    }
    .score-card:hover .score-tooltip {
        visibility: visible;
        opacity: 1;
    }
    .score-value { font-size: 1.8rem; font-weight: 700; }
    .score-label { font-size: 0.75rem; color: #64748b; }

    .score-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #1e293b;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        width: 250px;
        text-align: left;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: opacity 0.2s, visibility 0.2s;
        margin-bottom: 8px;
    }
    .score-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #1e293b;
    }
    .score-tooltip strong { color: #93c5fd; }
    .score-tooltip a { color: #93c5fd; text-decoration: underline; }

    .sources-footer {
        margin-top: 2rem;
        padding: 1rem;
        background: #f1f5f9;
        border-radius: 8px;
    }
    .sources-footer h4 { margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #475569; }
    .sources-footer a {
        display: inline-block;
        margin: 0.2rem 0.3rem;
        padding: 0.3rem 0.6rem;
        background: white;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #3b82f6;
        text-decoration: none;
        border: 1px solid #e2e8f0;
    }
    .sources-footer a:hover { background: #eff6ff; border-color: #3b82f6; }

    .data-age {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    .cached-badge {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }

    .recent-company {
        padding: 0.5rem;
        border-radius: 6px;
        margin-bottom: 0.3rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    .recent-company:hover { background: #f1f5f9; }
    .recent-company-name { font-weight: 500; font-size: 0.9rem; }
    .recent-company-age { font-size: 0.75rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# Session state
if "company_profile" not in st.session_state:
    st.session_state.company_profile = None
if "raw_content" not in st.session_state:
    st.session_state.raw_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "investment_score" not in st.session_state:
    st.session_state.investment_score = None
if "diligence_questions" not in st.session_state:
    st.session_state.diligence_questions = []
if "current_domain" not in st.session_state:
    st.session_state.current_domain = ""
if "data_updated_at" not in st.session_state:
    st.session_state.data_updated_at = None
if "is_cached" not in st.session_state:
    st.session_state.is_cached = False


def normalize_url(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.rstrip('/')


def get_short_path(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return "homepage"
    return path.split('/')[-1] or path.split('/')[0]


def quick_scrape(url: str, timeout: int = 10) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    except Exception:
        return ""


def discover_pages(base_url: str, max_pages: int = 10) -> list[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        domain = extract_domain(base_url)
        found_urls = set([base_url])

        priority_paths = ['/about', '/team', '/careers', '/jobs', '/pricing', '/products',
                         '/solutions', '/company', '/investors', '/press', '/customers']

        for path in priority_paths:
            found_urls.add(f"https://{domain}{path}")

        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/') and not href.startswith('//'):
                full_url = f"https://{domain}{href.split('?')[0].split('#')[0]}"
                found_urls.add(full_url)

            if len(found_urls) >= max_pages * 2:
                break

        return list(found_urls)[:max_pages]
    except Exception:
        return [base_url]


def scrape_multiple(urls: list[str]) -> str:
    all_content = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(quick_scrape, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    all_content.append(f"=== {url} ===\n{content}\n")
            except Exception:
                pass
    return '\n\n'.join(all_content)


def generate_investment_score(profile) -> dict:
    """Generate investment scores with source citations."""

    # Build evidence list with sources
    evidence = []

    if profile.description.value:
        src = profile.description.source or "homepage"
        evidence.append(f"Description: {profile.description.value} [source: {src}]")

    if profile.industry.value:
        src = profile.industry.source or "homepage"
        evidence.append(f"Industry: {profile.industry.value} [source: {src}]")

    if profile.business_model.value:
        src = profile.business_model.source or "homepage"
        evidence.append(f"Business Model: {profile.business_model.value} [source: {src}]")

    if profile.tech_stack:
        for item in profile.tech_stack.items[:5]:
            src = item.get("source", "website")
            evidence.append(f"Tech: {item.get('value')} [source: {src}]")

    if profile.founders:
        for item in profile.founders.items[:3]:
            src = item.get("source", "website")
            evidence.append(f"Founder: {item.get('value')} [source: {src}]")

    if profile.competitive_advantages:
        for item in profile.competitive_advantages.items[:3]:
            src = item.get("source", "website")
            evidence.append(f"Advantage: {item.get('value')} [source: {src}]")

    if profile.potential_risks:
        for item in profile.potential_risks.items[:3]:
            src = item.get("source", "website")
            evidence.append(f"Risk: {item.get('value')} [source: {src}]")

    evidence_text = "\n".join(evidence)
    source_urls = profile.source_urls[:8] if profile.source_urls else []

    prompt = f"""You are a VC analyst. Score this company (1-10 scale) based ONLY on the evidence provided.

Company: {get_company_name(profile)}
Funding Stage: {profile.funding_stage.value or 'Unknown'}

EVIDENCE (with sources):
{evidence_text}

AVAILABLE SOURCES:
{chr(10).join(source_urls)}

For each score, cite which source(s) from the list above informed your assessment.

Return JSON with this structure:
{{
  "market_opportunity": {{"score": 8, "reasoning": "...", "sources": ["url1", "url2"]}},
  "team_strength": {{"score": 7, "reasoning": "...", "sources": ["url1"]}},
  "product_differentiation": {{"score": 8, "reasoning": "...", "sources": ["url1"]}},
  "business_model": {{"score": 7, "reasoning": "...", "sources": ["url1"]}},
  "timing": {{"score": 8, "reasoning": "...", "sources": ["url1"]}},
  "overall": {{"score": 8, "reasoning": "...", "sources": ["url1", "url2"]}},
  "thesis_fit": "One sentence on fit for dev tools/infra thesis",
  "key_question": "Most important diligence question to answer"
}}

Be conservative - only score high if evidence supports it."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"overall": {"score": 5, "reasoning": "Unable to assess", "sources": []}}


def generate_diligence_questions(profile) -> list[str]:
    prompt = f"""VC partner preparing for founder meeting.

Company: {profile.name}
Description: {profile.description.value}
Industry: {profile.industry.value}
Risks: {', '.join(profile.potential_risks.values()) if profile.potential_risks else 'None'}

Generate 5 specific diligence questions. Return as JSON: {{"questions": [...]}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return data.get("questions", [])
    except:
        return []


def generate_deal_memo(profile, score) -> str:
    prompt = f"""Write a 300-word investment memo.

Company: {profile.name}
Description: {profile.description.value}
Industry: {profile.industry.value}
Business Model: {profile.business_model.value}
Founders: {', '.join(profile.founders.values()) if profile.founders else 'Unknown'}
Products: {', '.join(profile.key_products.values()) if profile.key_products else 'Unknown'}
Funding: {profile.funding_stage.value or 'Unknown'}
Advantages: {', '.join(profile.competitive_advantages.values()) if profile.competitive_advantages else 'None'}
Risks: {', '.join(profile.potential_risks.values()) if profile.potential_risks else 'None'}
Score: {score.get('overall', 'N/A')}/10

Structure: Overview, Thesis, Risks, Recommendation. Be direct."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


def chat_about_company(query: str, profile, domain: str) -> tuple[str, int]:
    """Chat about company using hybrid retrieval. Returns (response, chunks_used)."""
    # Use hybrid search (semantic + keyword) to get relevant context
    relevant_chunks = search_company_content(domain, query, n_results=5, hybrid=True)
    context = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""VC analyst assistant for {get_company_name(profile)}.
Industry: {profile.industry.value}
Model: {profile.business_model.value}
Products: {', '.join(profile.key_products.values()) if profile.key_products else 'Unknown'}

Relevant context from their website (retrieved via hybrid search):
{context}

Be concise and accurate. Only state what's supported by the context above.
If the context doesn't contain relevant information, say so."""
            },
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content, len(relevant_chunks)


def dict_to_profile(data: dict) -> CompanyProfile:
    """Convert stored dict back to CompanyProfile."""
    def to_cited_value(d):
        if isinstance(d, dict):
            return CitedValue(value=d.get("value", ""), source=d.get("source"))
        return CitedValue(value=str(d) if d else "", source=None)

    def to_cited_list(items):
        if isinstance(items, list):
            return CitedList(items=[
                {"value": item.get("value", ""), "source": item.get("source")}
                if isinstance(item, dict) else {"value": str(item), "source": None}
                for item in items
            ])
        return CitedList(items=[])

    return CompanyProfile(
        name=data.get("name", "Unknown"),
        description=to_cited_value(data.get("description", {})),
        founding_year=to_cited_value(data.get("founding_year", {})),
        founders=to_cited_list(data.get("founders", [])),
        headquarters=to_cited_value(data.get("headquarters", {})),
        industry=to_cited_value(data.get("industry", {})),
        business_model=to_cited_value(data.get("business_model", {})),
        tech_stack=to_cited_list(data.get("tech_stack", [])),
        key_products=to_cited_list(data.get("key_products", [])),
        target_customers=to_cited_value(data.get("target_customers", {})),
        funding_stage=to_cited_value(data.get("funding_stage", {})),
        notable_investors=to_cited_list(data.get("notable_investors", [])),
        team_size=to_cited_value(data.get("team_size", {})),
        hiring_signals=to_cited_list(data.get("hiring_signals", [])),
        competitive_advantages=to_cited_list(data.get("competitive_advantages", [])),
        potential_risks=to_cited_list(data.get("potential_risks", [])),
        source_urls=data.get("source_urls", [])
    )


def get_company_name(profile) -> str:
    """Safely get company name as string."""
    name = profile.name
    if isinstance(name, dict):
        return name.get("value", "Unknown")
    return str(name) if name else "Unknown"


def render_cited_list(cl: CitedList, icon: str = "•") -> None:
    if not cl or not cl.items:
        st.caption("Not found")
        return

    for item in cl.items:
        val = item.get("value", "")
        src = item.get("source")

        if src:
            st.markdown(f"{icon} {val} [[source]({src})]")
        else:
            st.markdown(f"{icon} {val}")


def load_company_from_cache(domain: str):
    """Load company from database into session state."""
    data = get_company(domain)
    if not data:
        return False

    st.session_state.company_profile = dict_to_profile(data["profile"])
    st.session_state.investment_score = data["score"]
    st.session_state.diligence_questions = data["questions"] or []
    st.session_state.current_domain = domain
    st.session_state.data_updated_at = data["updated_at"]
    st.session_state.is_cached = True
    st.session_state.messages = []

    if "deal_memo" in st.session_state:
        del st.session_state.deal_memo

    return True


def clear_session():
    """Clear current company from session."""
    st.session_state.company_profile = None
    st.session_state.investment_score = None
    st.session_state.diligence_questions = []
    st.session_state.current_domain = ""
    st.session_state.data_updated_at = None
    st.session_state.is_cached = False
    st.session_state.messages = []
    st.session_state.raw_content = ""
    if "deal_memo" in st.session_state:
        del st.session_state.deal_memo


# ============ UI ============

with st.sidebar:
    st.markdown("### 🎯 CompanyScout")
    st.caption("VC research with caching")
    st.divider()

    # API Key Settings
    with st.expander("⚙️ Settings", expanded=not bool(os.getenv("OPENAI_API_KEY"))):
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            st.success("✓ API key loaded from .env")
            use_custom = st.checkbox("Use different API key")
            if use_custom:
                custom_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    key="custom_api_key_input"
                )
                if custom_key:
                    st.session_state.openai_api_key = custom_key
                    st.success("✓ Using custom key")
        else:
            st.warning("No API key in .env")
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get your key at platform.openai.com",
                key="api_key_input"
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input
                st.success("✓ API key set")

    st.divider()

    url_input = st.text_input("Company URL", placeholder="linear.app", label_visibility="collapsed")

    # Scrape depth selector
    scrape_depth = st.radio(
        "Research depth",
        options=["Quick scan", "Standard", "Deep dive"],
        index=1,
        horizontal=True,
        help="Quick: 5 pages | Standard: 10 pages | Deep: 18 pages"
    )

    depth_to_pages = {"Quick scan": 5, "Standard": 10, "Deep dive": 18}
    max_pages = depth_to_pages[scrape_depth]

    col_analyze, col_refresh = st.columns(2)
    with col_analyze:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    with col_refresh:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True,
                               disabled=not st.session_state.company_profile,
                               help="Re-scrape and update data")

    # Status
    if st.session_state.company_profile:
        st.divider()
        status_text = f"✓ {get_company_name(st.session_state.company_profile)}"
        if st.session_state.is_cached:
            status_text += " (cached)"
        st.success(status_text)

        if st.session_state.data_updated_at:
            age = get_data_age(st.session_state.data_updated_at)
            st.caption(f"📅 Last updated: {age}")

        if st.button("✕ Clear", use_container_width=True):
            clear_session()
            st.rerun()

    # Recent companies
    st.divider()
    st.markdown("**Recent Research**")

    recent = get_recent_companies(limit=5)
    if recent:
        for company in recent:
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col1:
                if st.button(
                    f"📄 {company['name']}",
                    key=f"recent_{company['domain']}",
                    use_container_width=True
                ):
                    load_company_from_cache(company['domain'])
                    st.rerun()
            with col2:
                st.caption(get_data_age(company['updated_at']))
            with col3:
                if st.button("🗑️", key=f"delete_{company['domain']}", help="Delete"):
                    st.session_state.confirm_delete = company['domain']
                    st.rerun()

        # Delete confirmation dialog
        if "confirm_delete" in st.session_state and st.session_state.confirm_delete:
            domain_to_delete = st.session_state.confirm_delete
            st.warning(f"Delete **{domain_to_delete}**?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete", type="primary", use_container_width=True):
                    delete_company(domain_to_delete)
                    if st.session_state.current_domain == domain_to_delete:
                        clear_session()
                    st.session_state.confirm_delete = None
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete = None
                    st.rerun()
    else:
        st.caption("No research history yet")

    st.divider()
    st.caption("Built for Category Ventures")


# Initialize OpenAI client
client = get_openai_client()

# Analysis flow
should_analyze = analyze_btn and url_input
should_refresh = refresh_btn and st.session_state.current_domain

if should_analyze or should_refresh:
    # Check for API key
    if not client:
        st.error("⚠️ Please enter your OpenAI API key in Settings to continue.")
        st.stop()
    if should_refresh:
        url = normalize_url(st.session_state.current_domain)
        domain = st.session_state.current_domain
        force_refresh = True
    else:
        url = normalize_url(url_input)
        domain = extract_domain(url)
        force_refresh = False

    # Check cache first (unless refreshing)
    if not force_refresh and company_exists(domain):
        with st.spinner("Loading from cache..."):
            if load_company_from_cache(domain):
                st.rerun()

    # Fresh scrape
    with st.status("Researching company...", expanded=True) as status:
        st.write("🔍 Discovering pages...")
        pages = discover_pages(url, max_pages=max_pages)

        st.write(f"📥 Scraping {len(pages)} pages...")
        content = scrape_multiple(pages)
        st.session_state.raw_content = content

        if not content or len(content) < 500:
            st.error("Could not scrape website. It may be blocking automated requests.")
            st.stop()

        st.write("🧠 Analyzing with citations...")
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        profile = analyze_company(content, url, api_key=api_key)
        st.session_state.company_profile = profile

        st.write("📊 Generating scores...")
        score = generate_investment_score(profile)
        st.session_state.investment_score = score

        st.write("❓ Preparing questions...")
        questions = generate_diligence_questions(profile)
        st.session_state.diligence_questions = questions

        st.write("💾 Saving to database...")
        storage_stats = save_company(
            domain=domain,
            name=get_company_name(profile),
            url=url,
            profile_dict=profile_to_dict(profile),
            score=score,
            questions=questions,
            source_urls=profile.source_urls,
            raw_content=content
        )

        # Show storage confirmation
        vectordb_info = storage_stats.get("vectordb", {})
        chunk_count = vectordb_info.get("chunks", 0)
        st.write(f"✅ Saved to SQLite + ChromaDB ({chunk_count} chunks embedded)")

        st.session_state.current_domain = domain
        st.session_state.data_updated_at = None  # Will show "Just now"
        st.session_state.is_cached = False
        st.session_state.messages = []
        if "deal_memo" in st.session_state:
            del st.session_state.deal_memo

        status.update(label="Analysis complete", state="complete", expanded=False)

    st.rerun()


# Display results
if st.session_state.company_profile:
    profile = st.session_state.company_profile
    score = st.session_state.investment_score or {}

    # Header with cache indicator
    desc_text = profile.description.value if profile.description else ""
    cache_badge = '<span class="cached-badge">CACHED</span>' if st.session_state.is_cached else ''

    st.markdown(f"""
    <div class="company-header">
        <h1>{get_company_name(profile)} {cache_badge}</h1>
        <p>{desc_text}</p>
        <div class="stats-row">
            <span class="stat-chip">📍 {profile.headquarters.value or 'Unknown'}</span>
            <span class="stat-chip">🏢 {profile.industry.value}</span>
            <span class="stat-chip">💰 {profile.funding_stage.value or 'Unknown'}</span>
            <span class="stat-chip">👥 {profile.team_size.value or 'Unknown'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main grid
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown('<p class="section-header">Investment Scorecard</p>', unsafe_allow_html=True)

        def get_score_value(score_data):
            """Extract score value from nested or flat structure."""
            if isinstance(score_data, dict):
                return score_data.get("score", "?")
            return score_data

        def get_score_details(score_data):
            """Extract reasoning and sources from score."""
            if isinstance(score_data, dict):
                return score_data.get("reasoning", ""), score_data.get("sources", [])
            return "", []

        score_cols = st.columns(3)
        scores_data = [
            ("Market", "market_opportunity"),
            ("Team", "team_strength"),
            ("Product", "product_differentiation"),
            ("Model", "business_model"),
            ("Timing", "timing"),
            ("Overall", "overall"),
        ]

        for i, (label, key) in enumerate(scores_data):
            with score_cols[i % 3]:
                score_entry = score.get(key, "?")
                v = get_score_value(score_entry)
                reasoning, sources = get_score_details(score_entry)

                try:
                    v_int = int(v)
                    color = "#10b981" if v_int >= 7 else "#f59e0b" if v_int >= 5 else "#ef4444"
                except:
                    color = "#6b7280"
                    v_int = v

                # Build tooltip content
                tooltip_reasoning = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                tooltip_sources = ""
                if sources:
                    source_links = ", ".join([f'<a href="{s}" target="_blank">source</a>' for s in sources[:2]])
                    tooltip_sources = f"<br><strong>Sources:</strong> {source_links}"

                # Score card with hover tooltip
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-value" style="color:{color};">{v_int}</div>
                    <div class="score-label">{label}</div>
                    <div class="score-tooltip">
                        <strong>{label}</strong><br>
                        {tooltip_reasoning}
                        {tooltip_sources}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if score.get("thesis_fit"):
            st.info(f"**Thesis Fit:** {score.get('thesis_fit')}")
        if score.get("key_question"):
            st.warning(f"**Key Question:** {score.get('key_question')}")

    with col2:
        st.markdown('<p class="section-header">Competitive Advantages</p>', unsafe_allow_html=True)
        if profile.competitive_advantages:
            render_cited_list(profile.competitive_advantages, "✅")
        else:
            st.caption("No clear advantages identified")

        st.markdown('<p class="section-header">Risks & Concerns</p>', unsafe_allow_html=True)
        if profile.potential_risks:
            render_cited_list(profile.potential_risks, "⚠️")
        else:
            st.caption("No major risks identified")

    with col3:
        st.markdown('<p class="section-header">Key Details</p>', unsafe_allow_html=True)

        if profile.founders:
            st.markdown("**Founders**")
            render_cited_list(profile.founders, "•")

        if profile.key_products:
            st.markdown("**Products**")
            render_cited_list(profile.key_products, "•")

        if profile.tech_stack:
            st.markdown("**Tech Stack**")
            techs = []
            for item in profile.tech_stack.items[:6]:
                val = item.get("value", "")
                src = item.get("source")
                if src:
                    techs.append(f"[{val}]({src})")
                else:
                    techs.append(val)
            st.markdown(" • ".join(techs))

        if profile.notable_investors:
            st.markdown("**Investors**")
            investors = []
            for item in profile.notable_investors.items[:4]:
                val = item.get("value", "")
                src = item.get("source")
                if src:
                    investors.append(f"[{val}]({src})")
                else:
                    investors.append(val)
            st.markdown(", ".join(investors))

    st.divider()

    # Questions + Actions
    col_q, col_a = st.columns([2, 1])

    with col_q:
        st.markdown('<p class="section-header">Diligence Questions</p>', unsafe_allow_html=True)
        questions = st.session_state.diligence_questions
        if questions:
            for i, q in enumerate(questions[:5], 1):
                st.markdown(f"{i}. {q}")
        else:
            st.caption("No questions generated")

    with col_a:
        st.markdown('<p class="section-header">Actions</p>', unsafe_allow_html=True)

        if st.button("📝 Generate Deal Memo", use_container_width=True):
            with st.spinner("Writing memo..."):
                memo = generate_deal_memo(profile, score)
            st.session_state.deal_memo = memo
            st.rerun()

        if "deal_memo" in st.session_state and st.session_state.deal_memo:
            st.download_button(
                "📥 Download Memo",
                st.session_state.deal_memo,
                file_name=f"{get_company_name(profile).lower().replace(' ', '_')}_memo.md",
                mime="text/markdown",
                use_container_width=True
            )

        json_data = json.dumps(profile_to_dict(profile), indent=2)
        st.download_button(
            "📋 Export JSON",
            json_data,
            file_name=f"{get_company_name(profile).lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Memo display
    if "deal_memo" in st.session_state and st.session_state.deal_memo:
        st.divider()
        st.markdown('<p class="section-header">Investment Memo</p>', unsafe_allow_html=True)
        st.markdown(st.session_state.deal_memo)

    # Sources footer
    if profile.source_urls:
        st.markdown(f"""
        <div class="sources-footer">
            <h4>📚 Sources ({len(profile.source_urls)} pages analyzed)</h4>
            {''.join(f'<a href="{url}" target="_blank">{get_short_path(url)}</a>' for url in profile.source_urls[:10])}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Chat
    st.markdown('<p class="section-header">💬 Ask about this company</p>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response, chunks_used = chat_about_company(
                    prompt, profile, st.session_state.current_domain
                )
            st.markdown(response)
            st.caption(f"🔍 Retrieved {chunks_used} chunks via hybrid search")

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <h1 style="font-size:2.5rem; margin-bottom:0.5rem;">🎯 CompanyScout</h1>
        <p style="font-size:1.2rem; color:#64748b; margin-bottom:2rem;">VC-grade research with smart caching</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Investment Scorecard
        AI-generated scores across 6 dimensions with thesis fit analysis.
        """)

    with col2:
        st.markdown("""
        ### 💾 Smart Caching
        Research is saved locally. Instant reload for previously analyzed companies.
        """)

    with col3:
        st.markdown("""
        ### 🔗 Cited Claims
        Every fact links back to its source page for verification.
        """)

    st.divider()

    st.markdown("### Try these companies")
    examples = ["linear.app", "supabase.com", "posthog.com", "railway.app", "temporal.io"]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            st.code(ex, language=None)
