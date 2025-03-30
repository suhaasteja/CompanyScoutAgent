from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def retrieve_results(query):
    print(query)
    # query = "What's the Quiller?"
    results = vector_store.similarity_search_with_relevance_scores(query, k=10)  # Get top 3 matches


    for doc, score in results:
        st.write(f"Score: {score}\n Url: {doc.metadata['url']}\nTitle: {doc.metadata['title']}\nContent: {doc.page_content[:20]}\n")
        print(f"Score: {score}\n Url: {doc.metadata['url']}\nTitle: {doc.metadata['title']}\nContent: {doc.page_content[:20]}\n")


query = st.text_input("Ask a question:")

# Trigger results retrieval when a query is entered
if query:
    retrieve_results(query)