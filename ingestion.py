"""Injecting the LangChain documentation from the langchain-docs directory
ReadTheDocsLoader - Tool that helps to build documentation for GitHUb repositories"""
# Importing required libraries
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    inject_docs()
