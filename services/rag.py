import json
import numpy as np
from services.llm import embed_text

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return float(np.dot(a, b) / denom)

def build_contextual_query(store, company, user, question):
    last = store.recent(company, user, 4)
    hist = "\n".join([f"{m['role']}: {m['content']}" for m in last])
    return f"HISTÓRICO RELEVANTE:\n{hist}\n\nPERGUNTA ATUAL:\n{question}\n\nInterprete a intenção principal."

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