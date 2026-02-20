# 🚀 Multi-PDF RAG Chatbot 🤖📄

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload multiple PDF documents and ask questions across them.

The system retrieves relevant information from documents and generates accurate, context-aware answers using a Large Language Model (Llama 3.3 70B via Groq).

---

## 🌐 Live Demo

👉 (Add your Streamlit Cloud link here after deployment)

---

## ✨ Features

- 📄 Upload multiple PDF files
- 🔍 Semantic search across all documents
- 🧠 LLM-powered answers using Groq (Llama 3.3 70B)
- 📌 Displays source file names for answers
- ⚡ Fast and interactive Streamlit UI
- 🔄 In-memory processing (Streamlit Cloud compatible)

---

## 🧠 How It Works

1. Upload PDFs via Streamlit UI  
2. Documents are loaded and split into chunks  
3. Embeddings are generated using Sentence Transformers  
4. Stored in Chroma vector database (in-memory)  
5. User query → relevant chunks retrieved  
6. LLM generates answer using retrieved context  
7. Sources are displayed for transparency  

---

## 🛠 Tech Stack

- **LangChain**
- **Chroma Vector DB**
- **HuggingFace Embeddings** (`sentence-transformers`)
- **Groq LLM** (Llama 3.3 70B)
- **Streamlit**
- **PyPDF Loader**

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/your-username/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
