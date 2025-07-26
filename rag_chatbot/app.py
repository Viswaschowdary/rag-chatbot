import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Flan-T5 RAG Chatbot", page_icon="🧠")

# Sidebar
st.sidebar.title("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
clear = st.sidebar.button("🧹 Clear Chat")

if clear:
    st.session_state.memory.clear()
    st.session_state.chat_history = []

# Load PDF
if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()
else:
    docs = []

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Embeddings & VectorStore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# QA Chain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=memory)

# UI
st.title("🤖 Flan-T5 RAG Chatbot (FAISS)")
st.write("Ask a question based on your uploaded PDF.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:")

if query:
    result = qa.invoke({"question": query})
    st.session_state.chat_history.append((query, result["answer"]))

for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
