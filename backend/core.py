# Importing required libraries
import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)


# Function to run the LLM
def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name="langchain-docs-index", embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Creating the question and answer chain
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )  # Stuff method is simply to that the context and plug it into the query

    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
