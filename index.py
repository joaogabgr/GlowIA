import os
import re
import json
import glob
import time
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------
# CONFIGURAÇÃO GEMINI
# ---------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash-lite"

# ---------------------------------------
# TOKENIZER GLOBAL
# ---------------------------------------
ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------
# LÊ PERSONA DO ARQUIVO
# ---------------------------------------
with open("persona.txt", "r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()


# ---------------------------------------
# UTILITÁRIOS
# ---------------------------------------
def sanitize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\x00-\x1F]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def chunk_text(text: str, max_tokens=300):
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = ENCODING.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


def embed_text(text):
    """Cacheável futuramente."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query",
        request_options={"timeout": 10}
    )
    return np.array(result["embedding"], dtype=np.float32)


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return float(np.dot(a, b) / denom)


# ---------------------------------------
# HISTÓRICO (ARMAZENAMENTO EM DISCO COM LIMITAÇÃO)
# ---------------------------------------
class ConversationStore:
    MAX_HISTORY = 30  # limite de mensagens por usuário

    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._store = {}
        self._lock = threading.RLock()

    def _fname(self, company, user):
        safe = lambda s: re.sub(r"[^a-z0-9_-]", "_", s.lower())
        return os.path.join(self.base_dir, f"{safe(company)}__{safe(user)}.json")

    def append(self, company, user, role, content):
        with self._lock:
            fname = self._fname(company, user)
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []

            history.append({"role": role, "content": content})
            history = history[-self.MAX_HISTORY:]

            with open(fname, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    def recent(self, company, user, n):
        fname = self._fname(company, user)
        if not os.path.exists(fname):
            return []
        with open(fname, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history[-n:]

    def full(self, company, user):
        fname = self._fname(company, user)
        if not os.path.exists(fname): 
            return []
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear(self, company, user):
        fname = self._fname(company, user)
        if os.path.exists(fname):
            os.remove(fname)


# ---------------------------------------
# COMPANY STORE COM EMBEDDINGS PERSISTENTES
# ---------------------------------------
class CompanyDataStore:
    def __init__(self, data_dir, max_cache_size=12):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._cache = OrderedDict()
        self.max_cache = max_cache_size
        self._lock = threading.RLock()

    def _embedding_file(self, company_name):
        safe = re.sub(r"[^a-z0-9_-]", "_", company_name.lower())
        return os.path.join(self.data_dir, f"{safe}_embeddings.json")

    def _json_file(self, company_name):
        safe = re.sub(r"[^a-z0-9_-]", "_", company_name.lower())
        return os.path.join(self.data_dir, f"{safe}.json")

    def load_or_build_embeddings(self, company_name):
        """Carrega embeddings pré-calculados, ou gera se não existir."""
        json_fp = self._json_file(company_name)
        if not os.path.exists(json_fp):
            raise FileNotFoundError("Empresa não encontrada")

        emb_fp = self._embedding_file(company_name)

        # Se já existir embeddings — carrega
        if os.path.exists(emb_fp):
            with open(emb_fp, "r", encoding="utf-8") as f:
                return json.load(f)

        # Caso contrário, gera (APENAS UMA VEZ)
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = json.dumps(data, ensure_ascii=False)
        chunks = chunk_text(text)

        vecstore = []
        for i, chunk in enumerate(chunks):
            vec = embed_text(chunk).tolist()
            vecstore.append({
                "filename": os.path.basename(json_fp),
                "chunk_index": i,
                "text": chunk,
                "embedding": vec
            })

        with open(emb_fp, "w", encoding="utf-8") as f:
            json.dump(vecstore, f, ensure_ascii=False, indent=2)

        return vecstore

    def get_vector_store(self, company_name):
        key = company_name.lower()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            store = self.load_or_build_embeddings(company_name)

            self._cache[key] = store
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)

            return store


# ---------------------------------------
# SISTEMA RAG
# ---------------------------------------
def build_contextual_query(store: ConversationStore, company, user, question):
    last = store.recent(company, user, 4)
    hist = "\n".join([f"{m['role']}: {m['content']}" for m in last])
    return f"""
HISTÓRICO RELEVANTE:
{hist}

PERGUNTA ATUAL:
{question}

Interprete a intenção principal.
"""


def search(company_store, conv_store, company, user, query):
    context_query = build_contextual_query(conv_store, company, user, query)
    query_vec = embed_text(context_query)
    vector_store = company_store.get_vector_store(company)

    scored = []
    for item in vector_store:
        score = cosine_similarity(query_vec, np.array(item["embedding"]))
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:6]]


# ---------------------------------------
# GEMINI SAFE CALL (timeout + retry)
# ---------------------------------------
def generate_answer(prompt, retries=3):
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                request_options={"timeout": 15}
            )
            return response.text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)

# ---------------------------------------
# PRÉ-GERAÇÃO DE EMBEDDINGS AO INICIAR O SERVIDOR
# ---------------------------------------
def prebuild_all_embeddings():
    print("🔄 Gerando embeddings de todas as empresas...")

    base_path = "data/empresas"
    files = [f for f in os.listdir(base_path) if f.endswith(".json")]

    for file in files:
        company_name = file.replace(".json", "")
        print(f"➡️  Processando: {company_name}")

        try:
            # Gera e salva os embeddings
            company_store.load_or_build_embeddings(company_name)
            # Carrega no cache
            company_store.get_vector_store(company_name)

            print(f"✔ Embeddings gerados e armazenados para: {company_name}")
        except Exception as e:
            print(f"❌ Erro ao processar {company_name}: {e}")

    print("✅ Todos os embeddings prontos!")

# ---------------------------------------
# FastAPI
# ---------------------------------------
class ChatRequest(BaseModel):
    company_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class EndRequest(BaseModel):
    company_id: str
    user_id: str


app = FastAPI(title="IA Chat RAG (Escalável)")


conversation_store = ConversationStore(base_dir="data/conversations")
company_store = CompanyDataStore(data_dir="data/empresas")


@app.post("/chat")
def chat(req: ChatRequest):
    company = sanitize_text(req.company_id)
    user = sanitize_text(req.user_id)
    question = sanitize_text(req.question)

    conversation_store.append(company, user, "user", question)

    try:
        chunks = search(company_store, conversation_store, company, user, question)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Empresa não encontrada")

    context = "\n\n---\n\n".join(
        f"[{c['filename']} - Chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    history = conversation_store.full(company, user)

    full_prompt = f"""
{BASE_PROMPT}

===== HISTÓRICO =====
{json.dumps(history, ensure_ascii=False, indent=2)}

===== CONTEXTO (RAG) =====
{context}

===== PERGUNTA ATUAL =====
{question}

RESPOSTA:
    """

    answer = generate_answer(full_prompt)
    conversation_store.append(company, user, "assistant", answer)
    return {"answer": answer}


@app.delete("/chat")
def end_chat(req: EndRequest):
    conversation_store.clear(req.company_id, req.user_id)
    return {"status": "ended"}

@app.on_event("startup")
async def startup_event():
    prebuild_all_embeddings()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
