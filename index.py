import glob
import json
import os
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple

import google.generativeai as genai
import numpy as np
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").strip().lower()
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def chunk_text(text, max_tokens=300):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def sanitize_text(s: str):
    s = s.strip()
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def embed_text(text):
    if AI_PROVIDER == "gemini":
        embedding = genai.embed_content(
            model="models/text-embedding-004", content=text, task_type="retrieval_query"
        )
        return np.array(embedding["embedding"], dtype=np.float32)
    elif AI_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    else:
        raise ValueError("AI_PROVIDER inválido. Use 'gemini' ou 'openai'.")


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class ConversationStore:
    def __init__(self):
        self._store: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        self._lock = threading.RLock()

    def append(self, company_id: str, user_id: str, role: str, content: str):
        key = (company_id, user_id)
        with self._lock:
            self._store.setdefault(key, []).append({"role": role, "content": content})

    def recent(self, company_id: str, user_id: str, n: int = 4) -> List[Dict[str, str]]:
        key = (company_id, user_id)
        with self._lock:
            return self._store.get(key, [])[-n:]

    def clear(self, company_id: str, user_id: str):
        key = (company_id, user_id)
        with self._lock:
            if key in self._store:
                del self._store[key]


class CompanyDataStore:
    def __init__(self, data_dir: str, max_cache_size: int = 8):
        self.data_dir = data_dir
        self._index: Dict[str, str] = {}
        self._cache: OrderedDict[str, List[Dict]] = OrderedDict()
        self._lock = threading.RLock()
        self.max_cache_size = max_cache_size
        self._built = False

    def _normalize(self, s: str) -> str:
        return sanitize_text(s).lower()

    def _build_index(self):
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            base = os.path.splitext(os.path.basename(fp))[0]
            self._index[base] = fp
            empresa = data.get("empresa", {})
            nome = empresa.get("nome") or ""
            cnpj = empresa.get("cnpj") or ""
            contatos = empresa.get("contatos", {})
            site = contatos.get("site") or ""
            if nome:
                self._index[self._normalize(nome)] = fp
            if cnpj:
                self._index[self._normalize(cnpj)] = fp
            if site:
                self._index[self._normalize(site)] = fp
        self._built = True

    def _evict_if_needed(self):
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def get_vector_store(self, company_id: str) -> List[Dict]:
        key = self._normalize(company_id)
        with self._lock:
            if not self._built:
                self._build_index()
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            fp = self._index.get(key)
            if not fp:
                raise FileNotFoundError("Dados da empresa não encontrados")
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, ensure_ascii=False)
            chunks = chunk_text(text)
            vector_store: List[Dict] = []
            for i, chunk in enumerate(chunks):
                vec = embed_text(chunk).tolist()
                vector_store.append(
                    {
                        "filename": os.path.basename(fp),
                        "chunk_index": i,
                        "text": chunk,
                        "embedding": vec,
                    }
                )
            self._cache[key] = vector_store
            self._cache.move_to_end(key)
            self._evict_if_needed()
            return vector_store


conversation_store = ConversationStore()
company_store = CompanyDataStore(data_dir=os.path.join(os.getcwd(), "data"))


def build_contextual_query(company_id: str, user_id: str, query: str):
    last_messages = conversation_store.recent(company_id, user_id, n=4)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in last_messages])
    contextual_query = f"""
HISTÓRICO RELEVANTE:
{history_text}

PERGUNTA ATUAL:
{query}

Interprete a intenção principal da conversa.
"""
    return contextual_query


def search(company_id: str, user_id: str, query: str, top_k: int = 6):
    contextual_query = build_contextual_query(company_id, user_id, query)
    query_vec = embed_text(contextual_query)
    vector_store = company_store.get_vector_store(company_id)
    scored = []
    for item in vector_store:
        score = cosine_similarity(query_vec, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]


def generate_answer(full_prompt: str) -> str:
    if AI_PROVIDER == "gemini":
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(full_prompt)
        return response.text
    elif AI_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    else:
        raise ValueError("AI_PROVIDER inválido. Use 'gemini' ou 'openai'.")


class ChatRequest(BaseModel):
    company_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)


class EndRequest(BaseModel):
    company_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


app = FastAPI(
    title="IA Chat Service",
    description="Serviço FastAPI para conversas com RAG por empresa e usuário",
    version="1.0.0",
)


@app.post("/chat")
def chat(req: ChatRequest):
    company_id = sanitize_text(req.company_id)
    user_id = sanitize_text(req.user_id)
    question = sanitize_text(req.question)
    if not company_id or not user_id or not question:
        raise HTTPException(status_code=400, detail="Campos obrigatórios inválidos")
    try:
        company_store.get_vector_store(company_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Empresa não encontrada")
    conversation_store.append(company_id, user_id, "user", question)
    results = search(company_id, user_id, question, top_k=6)
    context = "\n\n---\n\n".join(
        [f"[{r['filename']} - Chunk {r['chunk_index']}]\n{r['text']}" for r in results]
    )
    full_prompt = f"""
REGRAS DO ASSISTENTE (ATENDENTE HUMANO, FORMAL, EDUCADO E ESPECIALISTA EM VENDAS):

1. COMPORTAMENTO HUMANO  
   - Utilize o comportamento de um atendente humano profissional, educado, cordial e atencioso.  
   - Mantenha uma comunicação formal, respeitosa e clara.  
   - Nunca mencione IA, tecnologia, documentos ou funcionamento interno.  
   - Não escreva como robô; comunique-se de forma natural, fluida e próxima.

2. ESPECIALISTA EM VENDAS – ABORDAGEM PROFISSIONAL  
   - Você é um vendedor treinado para explicar serviços e conduzir clientes até a decisão.  
   - Utilize técnicas reais de vendas: rapport, conexão emocional, geração de valor, SPIN Selling, persuasão leve, urgência, autoridade e prova social (quando existir no contexto).  
   - Destaque benefícios reais e diferenciais da clínica.

3. BASEAR-SE EXCLUSIVAMENTE NOS DADOS FORNECIDOS  
   - Utilize somente o “CONTEXTO” e o “HISTÓRICO DA CONVERSA”.  
   - Nunca invente informações; caso algo não exista no material, diga:  
     “Informação não encontrada nos registros, mas posso ajudar com qualquer outra dúvida.”  

4. FOCO NO CONTEXTO DO MOMENTO  
   - Identifique o serviço ou produto que o cliente está discutindo.  
   - Continue o atendimento apenas sobre esse serviço.  
   - Nunca misture assuntos ou trocar para outro serviço sem o cliente pedir.

5. DETECÇÃO DE MOMENTO DE VALOR  
   - Sempre monitore a conversa para perceber quando:  
       a) O cliente já entendeu o serviço.  
       b) O cliente demonstrou interesse.  
       c) O cliente perguntou benefícios, valores, duração, vantagens ou resultados.  
   - Quando perceber esses sinais, o atendente deve entender que é **o momento ideal para oferecer um agendamento**.

6. OFERTA DE AGENDAMENTO (FECHAMENTO ELEGANTE)  
   Quando detectar que o cliente já recebeu informações suficientes ou demonstrou interesse, você deve iniciar um fechamento natural e profissional:

   - Convide para agendar uma data.  
   - Sugira horários livres (se existir no contexto).  
   - Utilize frases como:  
     “Se desejar, posso verificar uma data disponível para você.”  
     “Ficarei feliz em agendar seu atendimento.”  
     “Podemos reservar um horário para garantir sua vaga.”  
     “Prefere manhã, tarde ou noite?”  
   - NUNCA ofereça agendamento cedo demais; faça isso somente após perceber que o cliente já entendeu o serviço.

7. COMUNICAÇÃO CLARA, EMPÁTICA E ESTRATÉGICA  
   - Utilize frases curtas e objetivas.  
   - Use listas somente quando necessário.  
   - Sempre demonstre empatia, cordialidade e interesse real em ajudar.  
   - Mantenha o ritmo da conversa agradável, profissional e atencioso.

8. PROATIVIDADE INTELIGENTE  
   - Sempre que o cliente mostrar dúvida, explique com calma.  
   - Sempre que o cliente demonstrar interesse, avance.  
   - Sempre que o cliente sinalizar intenção de compra, ofereça o agendamento.  
   - Nunca pressione; ofereça sempre como uma opção elegante e respeitosa.

===== HISTÓRICO =====
{json.dumps(conversation_store.recent(company_id, user_id, n=50), indent=2, ensure_ascii=False)}

===== CONTEXTO (RAG) =====
{context}

===== PERGUNTA ATUAL =====
{question}

RESPOSTA:
"""
    try:
        answer = generate_answer(full_prompt)
    except Exception:
        raise HTTPException(status_code=500, detail="Falha ao gerar resposta")
    conversation_store.append(company_id, user_id, "assistant", answer)
    return {"answer": answer}


@app.delete("/chat")
def end_chat(req: EndRequest):
    company_id = sanitize_text(req.company_id)
    user_id = sanitize_text(req.user_id)
    if not company_id or not user_id:
        raise HTTPException(status_code=400, detail="Campos obrigatórios inválidos")
    conversation_store.clear(company_id, user_id)
    return {"status": "ended"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
