import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from config.settings import BASE_PROMPT, MONGO_URI, MONGO_DB, MONGO_COLLECTION_ENTERPRISE, EMBEDDINGS_DIR, CONVERSATIONS_DIR
from stores.conversation_store import ConversationStore
from stores.company_store import CompanyDataStore
from utils.text import sanitize_text, format_for_whatsapp
from services.rag import search
from services.llm import generate_answer
from services.prebuild import prebuild_all_embeddings
from services.whapi import send_text_message
from config.settings import WHAPI_TIMEOUT

app = FastAPI(title="IA Chat RAG (Escalável)")

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

conversation_store = ConversationStore(base_dir=CONVERSATIONS_DIR)
company_store = CompanyDataStore(
    mongo_uri=MONGO_URI,
    local_dir=EMBEDDINGS_DIR,
    db_name=MONGO_DB,
    col_enterprise=MONGO_COLLECTION_ENTERPRISE
)

class ChatRequest(BaseModel):
    company_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

class EndRequest(BaseModel):
    company_id: str
    user_id: str

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
    full_prompt = f"{BASE_PROMPT}\n\n===== HISTÓRICO =====\n{json.dumps(history, ensure_ascii=False, indent=2)}\n\n===== CONTEXTO (RAG) =====\n{context}\n\n===== PERGUNTA ATUAL =====\n{question}\n\nRESPOSTA:\n    "
    answer = generate_answer(full_prompt)
    formatted = format_for_whatsapp(answer)
    conversation_store.append(company, user, "assistant", formatted)
    try:
        send_text_message(to=user, body=formatted, timeout=WHAPI_TIMEOUT, logger=logger)
    except Exception as e:
        logger.error(f"Falha ao enviar mensagem via Whapi: {e}")
    return {"answer": formatted}

@app.delete("/chat")
def end_chat(req: EndRequest):
    conversation_store.clear(req.company_id, req.user_id)
    return {"status": "ended"}

@app.on_event("startup")
async def startup_event():
    try:
        prebuild_all_embeddings(company_store)
    except Exception as e:
        print(f"Falha ao pré-gerar embeddings local: {e}")

class SendRequest(BaseModel):
    to: str = Field(min_length=8)
    body: str = Field(min_length=1)

@app.post("/messages/text")
def send_message(req: SendRequest):
    try:
        result = send_text_message(to=req.to, body=req.body, timeout=WHAPI_TIMEOUT, logger=logger)
        return {"status": "sent", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))