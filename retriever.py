from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from crew_test import run_faang_research, run_linkedin_content_generation

import streamlit as st

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(model="gpt-4o-mini")

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = ""

st.title("Able AI Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("View source context"):
                st.markdown(message["context"])


def format_chat_history(messages):
    formatted_history = ""
    for i, message in enumerate(messages):
        if message["role"] == "user":
            formatted_history += f"<query {i}>\n{message['content']}\n</query {i}>\n\n"
        elif message["role"] == "assistant":
            formatted_history += f"<response {i}>\n{message['content']}\n</response {i}>\n\n"
    return formatted_history

def retrieve_results(query):
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
        </query>
        
        <context>
        {context}
        </context>
    """
    
    llm_res = llm.invoke(PROMPT)
    return llm_res.content, context

if prompt := st.chat_input("Ask a question about Able"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    linkedin_check_prompt = f"""
    Determine if this prompt is asking for LinkedIn content generation or related to creating social media content.
    If it is asking for LinkedIn content, social media posts, or content marketing, respond with only 'True'.
    Otherwise, respond with only 'False'.
    
    Prompt: '{prompt}'
    """
    
    linkedin_response = llm.invoke(linkedin_check_prompt)
    is_linkedin_request = linkedin_response.content.strip().lower() == 'true'
    
    is_report_request = False
    if not is_linkedin_request:
        check_prompt = f"If this prompt is asking for a report on FAANG stocks, market analysis, or requires web search, respond with only 'True'. Otherwise, respond with only 'False': '{prompt}'"
        test_response = llm.invoke(check_prompt)
        is_report_request = test_response.content.strip().lower() == 'true'
    
    if is_linkedin_request:
        st.badge("LinkedIn Content Generation")
    elif is_report_request:
        st.badge("FAANG Market Research")
    else:
        st.badge("Using Knowledge Base")
    
    with st.spinner("Generating response..."):
        context = ""
        
        if is_linkedin_request:
            st.badge("Using CrewAI for LinkedIn content...")
            response = run_linkedin_content_generation()
        elif is_report_request:
            st.badge("Using CrewAI for FAANG research...")
            response = run_faang_research()
        else:
            st.badge("Using vector database...")
            response, context = retrieve_results(prompt)
        
    with st.chat_message("assistant"):
        st.markdown(response)
        if context:
            with st.expander("View source context"):
                st.markdown(context)
        
    st.session_state.messages.append({"role": "assistant", "content": response, "context": context})