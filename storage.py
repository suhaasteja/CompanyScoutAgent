"""
Storage module - SQLite for profiles, ChromaDB for embeddings
"""
import os
import json
import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# Paths
DB_DIR = Path(__file__).parent / "data"
DB_DIR.mkdir(exist_ok=True)
SQLITE_PATH = DB_DIR / "companies.db"
CHROMA_PATH = DB_DIR / "chroma_db"


def get_sqlite_connection():
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database schema."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            score_json TEXT,
            questions_json TEXT,
            source_urls_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def get_chroma_client():
    """Get ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_openai_ef(api_key: str = None):
    """Get OpenAI embedding function for ChromaDB."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=key,
        model_name="text-embedding-3-small"  # Cheaper, faster for storage
    )


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path.split('/')[0]
    return domain.lower().replace('www.', '')


def company_exists(domain: str) -> bool:
    """Check if company exists in database."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM companies WHERE domain = ?", (domain,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def get_company(domain: str) -> Optional[dict]:
    """Get company data from database."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT domain, name, url, profile_json, score_json, questions_json,
               source_urls_json, created_at, updated_at
        FROM companies WHERE domain = ?
    """, (domain,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "domain": row["domain"],
        "name": row["name"],
        "url": row["url"],
        "profile": json.loads(row["profile_json"]),
        "score": json.loads(row["score_json"]) if row["score_json"] else None,
        "questions": json.loads(row["questions_json"]) if row["questions_json"] else [],
        "source_urls": json.loads(row["source_urls_json"]) if row["source_urls_json"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def save_company(
    domain: str,
    name: str,
    url: str,
    profile_dict: dict,
    score: dict = None,
    questions: list = None,
    source_urls: list = None,
    raw_content: str = None
):
    """Save company to SQLite and content to ChromaDB."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    # Ensure all values are proper types for SQLite
    domain = str(domain) if domain else ""
    name = str(name) if name else "Unknown"
    url = str(url) if url else ""

    # Upsert into SQLite
    cursor.execute("""
        INSERT INTO companies (domain, name, url, profile_json, score_json,
                               questions_json, source_urls_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(domain) DO UPDATE SET
            name = excluded.name,
            url = excluded.url,
            profile_json = excluded.profile_json,
            score_json = excluded.score_json,
            questions_json = excluded.questions_json,
            source_urls_json = excluded.source_urls_json,
            updated_at = excluded.updated_at
    """, (
        domain,
        name,
        url,
        json.dumps(profile_dict),
        json.dumps(score) if score else None,
        json.dumps(questions) if questions else None,
        json.dumps(source_urls) if source_urls else None,
        now,
        now
    ))

    conn.commit()
    conn.close()

    # Store content chunks in ChromaDB for semantic search
    vectordb_stats = {"chunks": 0, "status": "skipped"}
    if raw_content:
        vectordb_stats = store_content_embeddings(domain, raw_content)

    return {
        "sqlite": "success",
        "vectordb": vectordb_stats
    }


def store_content_embeddings(domain: str, content: str) -> dict:
    """Store content chunks with embeddings in ChromaDB. Returns stats."""
    client = get_chroma_client()
    openai_ef = get_openai_ef()

    # Get or create collection for this domain
    collection_name = f"company_{domain.replace('.', '_').replace('-', '_')}"

    # Delete existing collection if refreshing
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

    # Split content into chunks
    chunks = split_content(content, chunk_size=1000, overlap=200)

    if not chunks:
        return {"chunks": 0, "collection": collection_name, "status": "empty"}

    # Add chunks to collection
    collection.add(
        ids=[f"{domain}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"domain": domain, "chunk_index": i} for i in range(len(chunks))]
    )

    return {
        "chunks": len(chunks),
        "collection": collection_name,
        "status": "success",
        "content_length": len(content)
    }


def split_content(content: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split content into overlapping chunks."""
    if not content:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]

        # Try to break at a newline or period
        if end < len(content):
            last_break = max(
                chunk.rfind('\n\n'),
                chunk.rfind('. '),
                chunk.rfind('\n')
            )
            if last_break > chunk_size // 2:
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def keyword_search(chunks: list[str], query: str, top_k: int = 5) -> list[tuple[int, float]]:
    """Simple BM25-style keyword search. Returns (index, score) tuples."""
    import re
    from collections import Counter

    # Tokenize query
    query_terms = set(re.findall(r'\w+', query.lower()))

    if not query_terms:
        return []

    scores = []
    for i, chunk in enumerate(chunks):
        chunk_terms = Counter(re.findall(r'\w+', chunk.lower()))
        # Simple term frequency score
        score = sum(chunk_terms.get(term, 0) for term in query_terms)
        # Boost exact phrase matches
        if query.lower() in chunk.lower():
            score += 10
        scores.append((i, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def search_company_content(domain: str, query: str, n_results: int = 5, hybrid: bool = True) -> list[str]:
    """Search company content using hybrid (semantic + keyword) search."""
    client = get_chroma_client()
    openai_ef = get_openai_ef()

    collection_name = f"company_{domain.replace('.', '_').replace('-', '_')}"

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=openai_ef
        )

        # Semantic search
        semantic_results = collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get more for hybrid merging
        )

        semantic_docs = semantic_results["documents"][0] if semantic_results["documents"] else []
        semantic_ids = semantic_results["ids"][0] if semantic_results["ids"] else []

        if not hybrid or not semantic_docs:
            return semantic_docs[:n_results]

        # Keyword search on the semantic results (lightweight hybrid)
        keyword_scores = keyword_search(semantic_docs, query, top_k=n_results)

        # Combine: prioritize docs that appear in keyword search
        keyword_indices = {idx for idx, _ in keyword_scores if _ > 0}

        # Reorder: keyword matches first, then remaining semantic results
        reranked = []
        seen = set()

        # Add keyword matches first (in their keyword-ranked order)
        for idx, score in keyword_scores:
            if score > 0 and idx not in seen:
                reranked.append(semantic_docs[idx])
                seen.add(idx)

        # Add remaining semantic results
        for i, doc in enumerate(semantic_docs):
            if i not in seen:
                reranked.append(doc)
                seen.add(i)

        return reranked[:n_results]

    except Exception as e:
        return []


def get_retrieval_stats(domain: str) -> dict:
    """Get stats about stored content for a domain."""
    client = get_chroma_client()
    collection_name = f"company_{domain.replace('.', '_').replace('-', '_')}"

    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        return {"chunks": count, "collection": collection_name, "status": "active"}
    except:
        return {"chunks": 0, "collection": collection_name, "status": "not_found"}


def get_recent_companies(limit: int = 10) -> list[dict]:
    """Get recently researched companies."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT domain, name, updated_at
        FROM companies
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "domain": row["domain"],
            "name": row["name"],
            "updated_at": row["updated_at"]
        }
        for row in rows
    ]


def delete_company(domain: str):
    """Delete company from both SQLite and ChromaDB."""
    # Delete from SQLite
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM companies WHERE domain = ?", (domain,))
    conn.commit()
    conn.close()

    # Delete from ChromaDB
    client = get_chroma_client()
    collection_name = f"company_{domain.replace('.', '_').replace('-', '_')}"
    try:
        client.delete_collection(collection_name)
    except:
        pass


def get_data_age(updated_at: str) -> str:
    """Get human-readable age of data."""
    if not updated_at:
        return "Unknown"

    try:
        updated = datetime.fromisoformat(updated_at)
        now = datetime.now()
        diff = now - updated

        if diff.days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                minutes = diff.seconds // 60
                return f"{minutes} min ago" if minutes > 1 else "Just now"
            return f"{hours}h ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            return updated.strftime("%b %d, %Y")
    except:
        return "Unknown"


# Initialize database on import
init_db()
