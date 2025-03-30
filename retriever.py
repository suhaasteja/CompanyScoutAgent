from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

import streamlit as st

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(model="gpt-4o-mini")

res = llm.invoke("hi")
print(res.content)

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def retrieve_results(query):
    # print(query)
    # query = "What's the Quiller?"
    results = vector_store.similarity_search_with_relevance_scores(query, k=5)  
    question = query
    context = ""
    
    for index, doc in enumerate(results):
        context += f'document {index}. {doc}'
    
    PROMPT = f"""
        You are an info bot for a company called Able. Answer this question based on given context retrieved form able's website
        <question>
        {question}
        <question>
        
        <context>
        {context}
        <context>
    """

    # print(PROMPT)
    
    llm_res = llm.invoke(PROMPT)
    
    st.write(llm_res.content)


query = st.text_input("Ask a question about Able")

# Trigger results retrieval when a query is entered
if query:
    retrieve_results(query)