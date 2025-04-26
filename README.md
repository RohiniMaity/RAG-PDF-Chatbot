
# RAG Chatbot with PDF

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot application that allows users to upload PDF documents and ask questions about their content. The chatbot leverages LangChain, Groq LLM, and Hugging Face embeddings for document retrieval and conversational AI.

---

## Features

- Upload one or multiple PDF files (supports only PDFs).
- Automatically processes PDFs by splitting into chunks with page metadata.
- Creates embeddings using Hugging Face's `all-MiniLM-L6-v2` model.
- Uses Groq's LLaMA 3 8B model for conversational question answering.
- Displays top matching document chunks for transparency.
- Supports chat history with download and clear options.
- Rate limits user input to avoid rapid querying.

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Groq API Key](https://www.groq.com/) (set as environment variable `GROQ_API_KEY`)
- `pip` package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

The key libraries include:

- streamlit
- langchain
- langchain_groq
- langchain_community
- python-dotenv
- pymupdf

### Environment Variables

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

- Upload PDF files using the sidebar uploader.
- Ask questions in the chat input about the uploaded document.
- View the chatbot's answers and top matching document chunks.
- Download chat history or clear the chat using sidebar buttons.

---

## File Structure

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

## Notes

- Only PDF files are supported.
- Temporary PDF files are saved in `/tmp` (Linux/macOS). Adjust as needed for Windows.
- Rate limiting enforces minimum 5 seconds between queries.
- Embeddings and vectorstore are persisted in `chroma_persist` directory.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [LangChain](https://langchain.com/)
- [Groq LLM](https://www.groq.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

---

## 🙋‍♀️ Author

**Rohini Maity**  
🌐 Passionate about GenAI, RAG systems, and building intuitive AI interfaces.

---