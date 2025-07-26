import os
import fitz  # PyMuPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


# PDF Loader
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load all PDFs and TXT files from the data/ folder
def load_documents(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            content = load_pdf(file_path)
        elif file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            continue
        docs.append(Document(page_content=content, metadata={"source": file_name}))
    return docs

# Process files
raw_docs = load_documents("data")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(raw_docs)

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save to vector store
Chroma.from_documents(chunks, embedding=embedding, persist_directory="db").persist()

print("✅ Vector store built from PDF and TXT files.")
