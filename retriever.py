from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

import streamlit as st

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(model="gpt-4o-mini")

# res = llm.invoke("hi")
# print(res.content)

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Able AI Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def format_chat_history(messages):
    formatted_history = ""
    for i, message in enumerate(messages):
        if message["role"] == "user":
            formatted_history += f"<query {i}>\n{message['content']}\n</query {i}>\n\n"
        elif message["role"] == "assistant":
            formatted_history += f"<response {i}>\n{message['content']}\n</response {i}>\n\n"
    return formatted_history

def retrieve_results(query):
    # print(query)
    # query = "What's the Quiller?"
    results = vector_store.similarity_search_with_relevance_scores(query, k=5)  
    context = ""
    
    for index, doc in enumerate(results):
        context += f'document {index}. {doc}'
    
    chat_history = format_chat_history(st.session_state.messages)
    
    PROMPT = f"""
        You are an info bot for a company called Able. Answer this query based on given context retrieved form able's website and chat history
        
        <chat_history>
        {chat_history}
        </chat_history>
        
        <query>
        {query}
        <query>
        
        <context>
        {context}
        <context>
    """
    
    llm_res = llm.invoke(PROMPT)
    return llm_res.content

if prompt := st.chat_input("Ask a question about Able"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        response = retrieve_results(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})