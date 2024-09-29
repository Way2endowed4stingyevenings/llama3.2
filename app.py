## Import necessary libraries
## Streamlit for building the web app
## Embedchain for the RAG functionality
## tempfile for creating temporary files and directories

import os
import tempfile
import streamlit as st
import embedchain
from embedchain import App
import base64
from streamlit_chat import message

## print(embedchain.__version__)
## Configure the Embedchain App
## For this application we will use Llama-3.2 using @ollama you can choose from OpenAI, anthropic or any other LLM.
## Select the vector database as the opensource chroma db (you are free to choose any other vector database of your choice)

def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {"model": "llama3.2:latest", "max_tokens": 250,
                    "temperature": 0.5, "stream": True, "base_url": 'http://localhost:11434'}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama3.2:latest", "base_url": 'http://localhost:11434'}},
        }
    )

    def display_pdf(file):
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = (f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" '
                       f'type="application/pdf"></iframe>')

        st.markdown(pdf_display, unsafe_allow_html=True)

# Set up the Streamlit App
# Streamlit lets you create user interface with just python code, for this app we will:
# Add a title to the app using 'st.title()'
# Add a description for the app using 'st.caption()'

st.title("chat with PDF using Llama 3.2")
st.caption("This app allows you to chat with a PDF using Llama 3.2 running locally with Ollama!")

db_path = tempfile.mkdtemp()
if "app" not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a sidebar for PDF upload and preview
# Users can upload and preview PDFs here.

with st.sidebar:
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

# Add the PDF to the knowledge base
# This processes the PDF and adds it to our ChromaDB vector database.

if st.button("Add PDF to Knowledge Base"):
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            st.session_state.app.add(f.name, data_type="pdf_file")
        os.remove(f.name)
    st.success(f"Added {pdf_file.name} to the knowledge base!")

# Set up the chat interface
# This displays the chat history and allows users to input questions.

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": "prompt"})
    messages(prompt, is_user=True)

# process user questions and display responses
# Add a button to clear chat history.

with st.spinner("Thinking..."):
    response = st.session_state.app.chat(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    messages(response)

if st.button("Clear  History"):
    st.session_state.messages = []
    
# Working Application demo using Streamlit
#Paste the above code in vscode or pycharm and run the following command: 'streamlit run http://app.py'