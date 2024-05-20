# Frontend integration with Streamlit library
# Importing required libraries
from typing import Set

from backend.core import run_llm
import time
import streamlit as st
from streamlit_chat import message

st.header("LangChain Documentation Helper Bot")

# Enter the prompt from the user
prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

# History of User Prompts
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

# Chat answer history
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


# Takes the list of the URLs and prints it with numbers and formats
def create_sources_string(sources_urls: Set[str]) -> str:
    if not sources_urls:
        return ""
    source_list = list(sources_urls)
    source_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(source_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        # time.sleep(3)
        # Get the response from the backend
        generated_response = run_llm(query=prompt)
        sources = [
            doc.metadata["source"] for doc in generated_response["source_documents"]
        ]


        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        # session state
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        # Streamlit chat package from streamlit library
        message(user_query, is_user=True)
        message(generated_response)
 