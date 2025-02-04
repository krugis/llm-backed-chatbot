import os
import faiss
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import torch
import pickle
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Setup Logging
LOG_FILE = "log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_message(level, message):
    """Logs messages with different levels (INFO, WARNING, ERROR)."""
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)

# ✅ Load Sentence Transformer (Quantized for Speed)
try:
    quantized_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embedder = SentenceTransformer(quantized_model, device="cpu")
    embedder = torch.quantization.quantize_dynamic(embedder, {torch.nn.Linear}, dtype=torch.qint8)
    log_message("info", "Sentence Transformer loaded successfully.")
except Exception as e:
    log_message("error", f"Error loading Sentence Transformer: {e}")

# ✅ Load LLaMA Model (Optimized for Speed)
try:
    llm = Llama(
        model_path="/home/endpoint11/endpoints/models/llama/llama-3.2-1b-q5_k_m.gguf",
        n_ctx=384,
        n_threads=4,
        n_batch=16,
        logits_all=False,
        use_mlock=True,
        use_mmap=True,
        mul_mat_q=True,
        use_kv_cache=True,
        f16_kv=True,
        verbose=True
    )
    log_message("info", "LLaMA model loaded successfully.")
except Exception as e:
    log_message("error", f"Error loading LLaMA model: {e}")

# ✅ FAISS Index Path & Data Path
INDEX_PATH = "./faiss_index_quantized.pkl"
DATA_PATH = "/home/endpoint11/knowledgebase/ecommend-related-files"
D = 384  # Embedding size

# ✅ Normalize embeddings before adding them to FAISS
def normalize(embeddings):
    faiss.normalize_L2(embeddings)
    return embeddings

# ✅ Load or Create FAISS Index
faiss_index = None
doc_chunks = {}

if os.path.exists(INDEX_PATH):
    try:
        with open(INDEX_PATH, "rb") as f:
            faiss_index, doc_chunks = pickle.load(f)
        log_message("info", "FAISS index loaded from file.")
    except Exception as e:
        log_message("error", f"Error loading FAISS index: {e}")
else:
    log_message("warning", "FAISS index not found, will be created after processing documents.")

# ✅ Extract Text from PDF & DOCX
def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        log_message("error", f"Error reading {file_path}: {e}")
    return text.strip()

# ✅ Chunk Text for Efficient Retrieval
def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ✅ Document Processing (Tokenized & Indexed)
def process_documents():
    global faiss_index, doc_chunks

    new_chunks = []
    new_ids = []
    embeddings = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if not file.endswith((".pdf", ".docx")) or file in doc_chunks:
            continue

        text = extract_text(file_path)
        if not text:
            continue  

        chunks = chunk_text(text, chunk_size=100)
        batch_embeddings = embedder.encode(chunks, batch_size=16, show_progress_bar=False, num_workers=4)
        
        embeddings.extend(batch_embeddings.tolist())  

        new_chunks.extend(chunks)
        new_ids.extend([(file, i) for i in range(len(chunks))])

    if new_chunks:
        embeddings = np.array(embeddings).astype(np.float32)

        if faiss_index is None:
            faiss_index = faiss.IndexHNSWFlat(D, 32)  # Use HNSW for fast retrieval
            faiss_index.hnsw.efConstruction = 20
            faiss_index.hnsw.efSearch = 16
            faiss_index.add(normalize(embeddings))  
        else:
            faiss_index.add(normalize(embeddings))  

        for i, chunk_id in enumerate(new_ids):
            doc_chunks[chunk_id] = new_chunks[i]

        with open(INDEX_PATH, "wb") as f:
            pickle.dump((faiss_index, doc_chunks), f)
        log_message("info", "Documents processed and FAISS index updated.")

# ✅ Optimized Retrieval
def retrieve(query, k=3):
    query_embedding = np.array(embedder.encode([query])).astype(np.float32)
    faiss.normalize_L2(query_embedding)  
    distances, indices = faiss_index.search(query_embedding, k)

    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(doc_chunks):
            retrieved_chunks.append(doc_chunks[list(doc_chunks.keys())[idx]])

    return retrieved_chunks

# ✅ Optimized Response Generation
def generate_response(query):
    retrieved_chunks = retrieve(query)
    
    # Reduce number of retrieved chunks to limit context
    context = "\n".join(retrieved_chunks[:3])  

    # Optimized Prompt
    prompt = (
        f"### Instruction: Answer concisely based on the given context.\n"
        f"### Context:\n{context}\n"
        f"### Question: {query}\n"
        f"### Answer:"
    )

    try:
        output = llm(prompt, max_tokens=40, stream=False)  
        response = output["choices"][0]["text"].strip()  
        log_message("info", f"Query: {query} | Response: {response}")
    except Exception as e:
        log_message("error", f"Error generating response: {e}")
        response = "Sorry, I encountered an error."

    return response

# ✅ Request Model
class QueryRequest(BaseModel):
    query: str

# ✅ FastAPI Endpoint (Takes JSON Input & Returns Only Response)
@app.post("/query")
def query_text(request: QueryRequest):
    return generate_response(request.query)

# ✅ Process Documents on Startup
process_documents()
log_message("info", "Server started and documents processed.")
