"""Injecting the LangChain documentation from the langchain-docs directory
ReadTheDocsLoader - Tool that helps to build documentation for GitHUb repositories"""
import os
from dotenv import load_dotenv, find_dotenv
# Importing required libraries
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import warnings

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT_REGION"))


# Function to inject the files from the langchain directories
def inject_docs() -> None:
    # Load the data
    loader = ReadTheDocsLoader(
        path="langchain-docs/langchain-docs/api.python.langchain.com/en/latest"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    # Splitting the text
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitting into {len(documents)} chunks")
    # Iterate over docs to retrieve metadat field "the source key"
    # And replacing it with https
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")

    # Embedding into OpenAI
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name="langchain-docs-index")
    print("Added to Pinecone VectorStore vectors")


if __name__ == "__main__":
    inject_docs()
