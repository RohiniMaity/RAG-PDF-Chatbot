
# 🤖 RAG Chatbot with PDF (Streamlit + LangChain + Groq)

This is a simple Retrieval-Augmented Generation (RAG) chatbot built using **Streamlit**, **LangChain**, **HuggingFace embeddings**, **Chroma vector store**, and **Groq's LLaMA3** LLM. It allows users to upload a PDF and ask context-aware questions based on the document.

---

## 📦 Features

- ✅ Upload and process PDF documents  
- ✅ Chunk text using recursive character splitting  
- ✅ Embed documents using `sentence-transformers/all-MiniLM-L6-v2`  
- ✅ Store and retrieve vectors using Chroma  
- ✅ Query documents using Groq's LLaMA3 model via LangChain  
- ✅ Interactive chat interface with memory using Streamlit  

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RohiniMaity/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory and add your **Groq API key**:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📁 Project Structure

```
📦rag-pdf-chatbot
 ┣ 📜RAG_Chatbot_Pdf.py  # Main Streamlit application
 ┣ 📜.env                # Environment file for API key
 ┣ 📜requirements.txt    # Python dependencies
 ┗ 📜README.md           # This file
```

---

## ▶️ Run the App

```bash
streamlit run RAG_Chatbot_Pdf.py
```

Then, open the app in your browser (typically at [http://localhost:8501](http://localhost:8501)).

---

## 📚 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Chroma](https://www.trychroma.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

---

## 📝 Usage Instructions

1. Use the sidebar to upload a **PDF file**.
2. Once uploaded, ask any question in the chat input box.
3. The bot will search for relevant content in the document and provide an accurate, concise response using the LLaMA3 model.

---

## 🛠️ Notes

- Only **PDF** files are supported at the moment.
- Chroma vector store is in-memory, meaning it resets when the app restarts.
- You must have a **Groq API key** to use the LLM.

---

## 📃 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♀️ Author

**Rohini Maity**  
🌐 Passionate about GenAI, RAG systems, and building intuitive AI interfaces.

---
