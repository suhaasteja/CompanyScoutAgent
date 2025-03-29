from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

query = "What's the Quiller?"
results = vector_store.similarity_search_with_relevance_scores(query, k=3)  # Get top 3 matches


for doc, score in results:
    print(f"Score: {score}\n Url: {doc.metadata['url']}\nTitle: {doc.metadata['title']}\nContent: {doc.page_content[:20]}\n")
