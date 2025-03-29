
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI()
# llm.invoke("Hello, world!")

import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma


load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

res = llm.invoke("Hello, world!")

print(res.content)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="able_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
from uuid import uuid4

from langchain_core.documents import Document

documents = []

import pandas as pd

path = "website_scrape.csv"

df = pd.read_csv(path)

from langchain_text_splitters import RecursiveCharacterTextSplitter


for index, row in df.iterrows():
    url = row['url']
    title = row['title']
    markdown = row['markdown']
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_text(markdown)
    
    if not all_splits:
        continue 
    
    for split in all_splits:
        
        documents.append(
            Document(
                page_content=split,
                metadata = {"url": url, "title": title},
                id=str(uuid4())
            )
        )
    
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)