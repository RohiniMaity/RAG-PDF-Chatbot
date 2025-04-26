import os
import time
from dotenv import load_dotenv

import streamlit as st                                               #create web applications
import streamlit.components.v1 as components

from langchain_groq import ChatGroq                                  #enables conversational AI models to interact with users using natural language
from langchain_core.output_parsers import StrOutputParser            #parses the output of a conversational AI model into a string format
from langchain_core.prompts import ChatPromptTemplate                #helps generate prompts for conversational AI models

from langchain_community.embeddings import HuggingFaceEmbeddings     #use of Hugging Face's pre-trained language models for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter   #splits text into smaller chunks based on character count
from langchain_community.document_loaders import PyMuPDFLoader       #loading of PDF documents into a format that can be processed by the LangChain library
from langchain.chains import RetrievalQA                             #creation of question-answering models using retrieval-based approaches
from langchain_community.vectorstores import Chroma                  #Chroma is a vector database that stores and retrieves embeddings

load_dotenv()

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="RAG Chatbot with PDF",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RAG Chatbot with PDF")

# ------------------------------
# Initialize Session State Variables
# ------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = ""

if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = 0

if 'history' not in st.session_state:
    st.session_state.history = ""

if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

# ------------------------------
# Helper Function: Process Uploaded PDFs
# ------------------------------
def process_uploaded_pdfs(uploaded_files):
    all_docs = []
    try:
        if len(uploaded_files) == 0:
            st.error("Uploaded file is empty. Please upload a valid PDF.")
            return None

        for uploaded_file in uploaded_files:
            if not uploaded_file.name.lower().endswith(".pdf"):
                st.error(f"Unsupported file type: {uploaded_file.name}. Only PDF files are supported.")
                continue

            st.success(f"File {uploaded_file.name} uploaded successfully!")
            st.session_state.pdf_uploaded = True
            st.session_state.pdf_path = uploaded_file.name  # for reference

            # Save uploaded PDF to a temporary file
            tmp_path = f"/tmp/{uploaded_file.name}"
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())

            # Load PDF using PyMuPDFLoader
            with st.spinner(f"Processing {uploaded_file.name}..."):
                loader = PyMuPDFLoader(tmp_path)
                documents = loader.load()

                # Add page number metadata for reference
                for i, doc in enumerate(documents):
                    doc.metadata["page_number"] = i + 1

                # Split documents into chunks for embedding
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                all_docs.extend(docs)

        return all_docs

    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
        return None

# ------------------------------
# Sidebar UI: PDF Upload & Controls
# ------------------------------
with st.sidebar:
    st.header("Tools")

    st.title("Only PDF files are supported.")
    uploaded_files = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        docs = process_uploaded_pdfs(uploaded_files)
        if docs:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            persist_dir = "chroma_persist"
            st.session_state.vectorstore = Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            st.session_state.vectorstore.persist()

    if st.session_state.messages:
        if st.button("Clear Chat"):
            st.session_state.messages.clear()
            st.session_state.pdf_uploaded = False
            st.session_state.vectorstore = None
            st.session_state.pdf_path = ""
            st.experimental_rerun()

        if st.download_button(
            label="Download Chat History (.txt)",
            data=st.session_state.history,
            file_name="chat_history.txt",
            mime="text/plain"
        ):
            st.success("Chat history downloaded successfully!")

# ------------------------------
# Display Chat History
# ------------------------------
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# ------------------------------
# Chat Input & Response Logic
# ------------------------------
prompt = st.chat_input("Ask something about the document...")

if prompt:
    current_time = time.time()
    time_diff = current_time - st.session_state.last_query_time

    if time_diff < 5:
        st.warning("You're sending messages too fast. Please wait a few seconds.")
    else:
        st.session_state.last_query_time = current_time

        # Display user's prompt
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Validate PDF upload and vectorstore availability
        if not st.session_state.pdf_uploaded or not st.session_state.vectorstore:
            st.error("Please upload a PDF first.")
        else:
            try:
                # Initialize LLM with prompt template
                groq_sys_prompt = ChatPromptTemplate.from_template(
                    """
                    You are very smart at everything, you always give the best,
                    the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                    Start the answer directly. No small talk please.
                    """
                )

                groq_chat = ChatGroq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    model_name="llama3-8b-8192"
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )

                with st.spinner("Thinking..."):
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    top_docs = result["source_documents"]

                # Show top matching chunks in sidebar expandable section
                with st.sidebar.expander("🔍 Top Matching Chunks", expanded=False):
                    for i, doc in enumerate(top_docs):
                        page_num = doc.metadata.get('page_number', 'N/A')
                        st.markdown(f"**Chunk {i+1} (Page {page_num}):**")
                        st.write(doc.page_content[:500])  # Preview first 500 characters

                # Stream assistant response word-by-word with small delay
                def stream_response(text):
                    for word in text.split(" "):
                        yield word + " "
                        time.sleep(0.03)

                with st.chat_message("assistant"):
                    st.write_stream(stream_response(response))

                # Save assistant's reply in session history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.history = "\n\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
                )

            except Exception as e:
                st.error("Something went wrong. Please try again later.")
                st.exception(e)  # Optional detailed error info for debugging
