import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# ---------------------
# Load ENV
# ---------------------
load_dotenv()

# ---------------------
# Embedding Model
# ---------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------
# LLM (Groq)
# ---------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ---------------------
# 1️⃣ Process Multiple PDFs (IN-MEMORY)
# ---------------------
def process_documents(uploaded_files):

    all_docs = []

    for file in uploaded_files:
        # Save temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Add metadata (file name)
        for doc in docs:
            doc.metadata["source"] = file.name

        all_docs.extend(docs)

        # Remove temp file
        os.remove(temp_path)

    # ---------------------
    # Split Documents
    # ---------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    # ✅ CRITICAL FIX (avoid empty embeddings error)
    if not chunks:
        raise ValueError("No text could be extracted from uploaded PDFs")

    # Optional debug
    print(f"Total chunks created: {len(chunks)}")

    # ---------------------
    # Create Vector DB (in-memory)
    # ---------------------
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vectordb


# ---------------------
# 2️⃣ QA Function
# ---------------------
def answer_question(vectordb, query):

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain({"query": query})

    answer = response["result"]

    # Extract unique source file names
    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in response["source_documents"]
    ]))

    return answer, sources
