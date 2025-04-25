#Setup UI
import streamlit as st                                               #create web applications

#Connect LLM
import os
from langchain_groq import ChatGroq                                  #enables conversational AI models to interact with users using natural language
from langchain_core.output_parsers import StrOutputParser            #parses the output of a conversational AI model into a string format
from langchain_core.prompts import ChatPromptTemplate                #helps generate prompts for conversational AI models
from dotenv import load_dotenv

#Integrate RAG
from langchain.embeddings import HuggingFaceEmbeddings               #use of Hugging Face's pre-trained language models for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter   #splits text into smaller chunks based on character count
from langchain_community.document_loaders import PyPDFLoader         #loading of PDF documents into a format that can be processed by the LangChain library
from langchain.indexes import VectorstoreIndexCreator                #creation of indexes for vector-based search
from langchain.chains import RetrievalQA                             #creation of question-answering models using retrieval-based approaches
from langchain_community.vectorstores import Chroma                  #Chroma is a vector database that stores and retrieves embeddings

load_dotenv()

st.title("RAG Chatbot with PDF")

#Setup session state variable
if 'messages' not in st.session_state:                                #session state variable to store chat messages
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:                             #session state variable to store vectorstore
    st.session_state.vectorstore = None
if 'pdf_uploaded' not in st.session_state:                            #session state variable to track if a PDF is uploaded
    st.session_state.pdf_uploaded = False
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = ""

# Sidebar for uploading PDF
with st.sidebar:
    st.header("📂 Load your document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

        st.session_state.pdf_uploaded = True
        st.session_state.pdf_path = pdf_path

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask something about the document...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not st.session_state.pdf_uploaded or not st.session_state.vectorstore:
        st.error("Please upload a PDF first.")
    else:
        try:
            # Create prompt and model
            groq_sys_prompt = ChatPromptTemplate.from_template("""
                You are very smart at everything, you always give the best,
                the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                Start the answer directly. No small talk please.""")

            groq_chat = ChatGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
                model_name="llama3-8b-8192"
            )

            # Create RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            result = qa_chain({"query": prompt})
            response = result["result"]

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {e}")