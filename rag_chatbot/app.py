import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Flan-T5 RAG Chatbot", page_icon="🤖")
st.title("🤖 Flan-T5 RAG Chatbot (with Memory & Sidebar)")

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.title("📁 Upload & Settings")

uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if st.sidebar.button("Clear Chat Memory"):
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.success("✅ Chat memory cleared!")

# ----------------------------- DOCUMENT LOADING -----------------------------
docs = []
if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

# ----------------------------- SPLIT & STORE -----------------------------
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
else:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

# ----------------------------- MEMORY -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ----------------------------- LLM & QA Chain -----------------------------
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=flan_pipeline)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=st.session_state.memory,
    return_source_documents=False
)

# ----------------------------- CHAT -----------------------------
query = st.text_input("Ask a question:")

if query:
    result = qa.invoke({"question": query})
    st.write("🤖 Bot:", result["answer"])
