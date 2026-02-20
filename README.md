# 🚀 Multi-PDF RAG Chatbot 🤖📄

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload multiple PDF documents and ask questions across them. The system retrieves relevant information and generates accurate, context-aware answers using a Large Language Model.

---

## 🌐 Live Demo

👉 **Try the app here:**  
https://multi-pdf-rag-ai-chatbot.streamlit.app/

---

## 🎯 Problem Statement

Traditional document search is limited to keyword matching and cannot understand context across multiple documents.

This project solves that by:

- Enabling **semantic search across multiple PDFs**
- Providing **context-aware answers using LLMs**
- Ensuring **transparency via source attribution**

👉 Goal: Build a **multi-document intelligent assistant**.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

- 🔍 **Retriever** → Finds relevant chunks from documents  
- 🧠 **Generator (LLM)** → Generates answer using retrieved context  

---

## ⚙️ How RAG Works (in this project)

User uploads PDFs  
↓  
Documents loaded & split into chunks  
↓  
Embeddings created (sentence-transformers)  
↓  
Stored in Chroma vector DB  
↓  
User asks question  
↓  
Top relevant chunks retrieved  
↓  
Groq LLM (Llama 3.3 70B) generates answer  
↓  
Answer + Source files displayed  

---

## 🏗️ Architecture

Streamlit UI  
↓  
Document Loader  
↓  
Text Splitter  
↓  
Embedding Model (MiniLM)  
↓  
Chroma Vector DB  
↓  
Retriever  
↓  
Groq LLM (Llama)  
↓  
Answer + Sources  

---

## ✨ Features

- 📄 Upload multiple PDF files  
- 🔍 Semantic search across documents  
- 🧠 LLM-powered answers (Groq - Llama 3.3 70B)  
- 📌 Source attribution (file names)  
- ⚡ Fast Streamlit UI  
- ☁️ Streamlit Cloud compatible  

---

## 🛠 Tech Stack

| Component | Tool |
|----------|------|
| UI | Streamlit |
| LLM | Groq (Llama 3.3 70B) |
| Framework | LangChain |
| Embeddings | Sentence Transformers |
| Vector DB | Chroma |
| PDF Loader | PyPDF |

---

## 📦 How to Run Locally

### 1️⃣ Clone repo

```bash
git clone https://github.com/your-username/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
