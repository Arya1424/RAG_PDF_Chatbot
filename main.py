from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import uuid
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime
import requests as http_requests

app = FastAPI(title="RAG Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384
print("Embedding model loaded!")

# HuggingFace API configuration (FREE!)
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: Get free token from huggingface.co

print("Using HuggingFace Inference API (Free) for answer generation")
print("Note: First few requests may be slow as model loads. Set HF_TOKEN env var for faster responses.")

# Storage directories
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_dbs")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# In-memory storage (use database in production)
subjects_db = {}
documents_db = {}

# Models
class Subject(BaseModel):
    name: str

class ChatRequest(BaseModel):
    subject_id: str
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    
    return chunks

def load_vector_db(subject_id: str):
    """Load FAISS index and metadata for a subject"""
    index_path = VECTOR_DB_DIR / f"{subject_id}.index"
    meta_path = VECTOR_DB_DIR / f"{subject_id}.meta"
    
    if not index_path.exists():
        return None, None
    
    index = faiss.read_index(str(index_path))
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return index, metadata

def save_vector_db(subject_id: str, index, metadata):
    """Save FAISS index and metadata for a subject"""
    index_path = VECTOR_DB_DIR / f"{subject_id}.index"
    meta_path = VECTOR_DB_DIR / f"{subject_id}.meta"
    
    faiss.write_index(index, str(index_path))
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

def generate_answer(question: str, context: str) -> str:
    """
    Generate answer using HuggingFace Inference API (FREE!)
    No local GPU needed, responses in 2-5 seconds
    """
    prompt = f"""<s>[INST] You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say "No information found in the subject documents."

Context:
{context[:1000]}

Question: {question}

Provide a concise answer based only on the context above. [/INST]"""

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        print("Calling HuggingFace API...")
        response = http_requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            },
            timeout=60
        )
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
                if answer:
                    print(f"✓ Generated answer: {answer[:100]}...")
                    return answer
            elif isinstance(result, dict) and "generated_text" in result:
                answer = result["generated_text"].strip()
                if answer:
                    print(f"✓ Generated answer: {answer[:100]}...")
                    return answer
            
            print(f"Unexpected API response format: {result}")
        
        elif response.status_code == 503:
            print("Model is loading on HuggingFace servers, please wait...")
            return "The AI model is currently loading. Please try again in 10-20 seconds."
        
        else:
            print(f"API Error {response.status_code}: {response.text}")
            
    except http_requests.exceptions.Timeout:
        print("Request timed out")
        return "The request timed out. Please try again."
    except Exception as e:
        print(f"Error calling HuggingFace API: {type(e).__name__}: {e}")
    
    # If we get here, use simple extractive answer
    print("Falling back to extractive answer...")
    return extract_answer_from_context(question, context)

def extract_answer_from_context(question: str, context: str) -> str:
    """
    Simple extractive answer - finds the most relevant sentence
    """
    # Split context into sentences
    sentences = [s.strip() for s in context.replace('\n', ' ').split('.') if s.strip()]
    
    # Find most relevant sentence (simple keyword matching)
    question_words = set(question.lower().split())
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences[:10]:  # Check first 10 sentences
        sentence_words = set(sentence.lower().split())
        score = len(question_words & sentence_words)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    if best_sentence:
        return best_sentence + "."
    else:
        return sentences[0] + "." if sentences else "No specific answer found in the documents."

# API Endpoints
@app.post("/subjects")
async def create_subject(subject: Subject):
    """Create a new subject"""
    subject_id = str(uuid.uuid4())
    subjects_db[subject_id] = {
        "id": subject_id,
        "name": subject.name,
        "created_at": datetime.now().isoformat(),
        "document_count": 0
    }
    return subjects_db[subject_id]

@app.get("/subjects")
async def list_subjects():
    """List all subjects"""
    return list(subjects_db.values())

@app.post("/subjects/{subject_id}/documents")
async def upload_documents(
    subject_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload documents to a subject"""
    if subject_id not in subjects_db:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Load or create vector database
    index, metadata = load_vector_db(subject_id)
    if index is None:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine similarity)
        metadata = {"chunks": [], "sources": []}
    
    uploaded_files = []
    
    for file in files:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{subject_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(str(file_path))
        elif file.filename.endswith('.txt'):
            text = extract_text_from_txt(str(file_path))
        else:
            continue
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
        
        # Add to FAISS index
        index.add(embeddings.astype('float32'))
        
        # Store metadata
        for chunk in chunks:
            metadata["chunks"].append(chunk)
            metadata["sources"].append(file.filename)
        
        uploaded_files.append(file.filename)
        
        # Update document count
        subjects_db[subject_id]["document_count"] += 1
    
    # Save vector database
    save_vector_db(subject_id, index, metadata)
    
    return {
        "subject_id": subject_id,
        "uploaded_files": uploaded_files,
        "total_chunks": len(metadata["chunks"])
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question to a subject"""
    print(f"\n=== CHAT REQUEST ===")
    print(f"Subject ID: {request.subject_id}")
    print(f"Question: {request.question}")
    
    if request.subject_id not in subjects_db:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Load vector database
    index, metadata = load_vector_db(request.subject_id)
    
    print(f"Index loaded: {index is not None}")
    if index:
        print(f"Total vectors in index: {index.ntotal}")
        print(f"Total chunks in metadata: {len(metadata.get('chunks', []))}")
    
    if index is None or index.ntotal == 0:
        print("ERROR: No vectors in database!")
        return ChatResponse(
            answer="No information found in the subject documents.",
            sources=[]
        )
    
    # Embed question
    question_embedding = embedding_model.encode(
        [request.question], 
        normalize_embeddings=True
    ).astype('float32')
    
    # Search similar chunks
    k = min(request.top_k, index.ntotal)
    distances, indices = index.search(question_embedding, k)
    
    print(f"Top {k} similarity scores: {distances[0]}")
    print(f"Top {k} chunk indices: {indices[0]}")
    
    # Relevance threshold (cosine similarity > 0.3)
    relevance_threshold = 0.2  # Lowered from 0.3
    
    # Collect relevant chunks
    relevant_chunks = []
    relevant_sources = []
    
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Checking chunk {idx}: similarity={dist:.4f}, threshold={relevance_threshold}")
        if dist > relevance_threshold:
            chunk_text = metadata["chunks"][idx]
            print(f"✓ RELEVANT: {chunk_text[:100]}...")
            relevant_chunks.append(chunk_text)
            relevant_sources.append(metadata["sources"][idx])
        else:
            print(f"✗ Not relevant enough")
    
    if not relevant_chunks:
        print("ERROR: No chunks passed relevance threshold!")
        return ChatResponse(
            answer="No information found in the subject documents.",
            sources=[]
        )
    
    # Generate answer using context
    context = "\n\n".join(relevant_chunks)
    
    # Limit context size
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    print(f"Context to send to LLM:\n{context[:200]}...\n")
    
    # Use HuggingFace API for answer generation
    answer = generate_answer(request.question, context)
    
    print(f"Final answer: {answer[:200]}...")
    
    return ChatResponse(
        answer=answer,
        sources=list(set(relevant_sources))
    )

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API is running",
        "model": "HuggingFace Mistral-7B (Free API, No Local GPU Required)"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "HuggingFace Mistral-7B Instruct (Free API)",
        "subjects_count": len(subjects_db),
        "hf_token_set": bool(HF_API_TOKEN)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)