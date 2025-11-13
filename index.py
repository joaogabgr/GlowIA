import json
import numpy as np
import os
import glob
import tiktoken
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configurar Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Ativar logs
print_selected_chunks = True

# Histórico da conversa
conversation_history = []


##########################################
# 1. CHUNK DE TEXTOS
##########################################

def chunk_text(text, max_tokens=300):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


##########################################
# 2. EMBEDDINGS GEMINI
##########################################

def embed_text(text):
    embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(embedding["embedding"], dtype=np.float32)


##########################################
# 3. CARREGAR JSON INFO
##########################################

def load_documents():
    docs = []
    try:
        with open("info.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # Cada seção do JSON vira um documento separado
        for key, content in data.items():
            docs.append({
                "filename": f"info.json - {key}",
                "content": {key: content}
            })

    except FileNotFoundError:
        print("❌ Arquivo info.json não encontrado!")

    return docs


##########################################
# 4. GERAR E SALVAR EMBEDDINGS
##########################################

def build_vector_store():
    vector_store = []

    docs = load_documents()

    for doc in docs:
        text = json.dumps(doc["content"], ensure_ascii=False, indent=2)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            vec = embed_text(chunk).tolist()

            vector_store.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk,
                "embedding": vec
            })

    with open("embeddings_store.json", "w", encoding="utf-8") as f:
        json.dump(vector_store, f, indent=2, ensure_ascii=False)

    print("✔ Embeddings criados e salvos!")


##########################################
# 5. BUSCA CONTEXTUAL (USANDO HISTÓRICO)
##########################################

def build_contextual_query(query):
    """
    Une o histórico com a pergunta atual,
    criando uma query mais inteligente e contextual.
    """

    last_messages = conversation_history[-4:]  # últimas interações

    history_text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in last_messages
    ])

    contextual_query = f"""
HISTÓRICO RELEVANTE:
{history_text}

PERGUNTA ATUAL:
{query}

Interprete a intenção principal da conversa.
"""

    return contextual_query


##########################################
# 6. SIMILARIDADE COSENO + RAG
##########################################

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return np.dot(a, b) / denom


def search(query, top_k=6):
    contextual_query = build_contextual_query(query)

    query_vec = embed_text(contextual_query)

    with open("embeddings_store.json", "r", encoding="utf-8") as f:
        vector_store = json.load(f)

    scored = []
    for item in vector_store:
        score = cosine_similarity(query_vec, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Debug (ver chunks selecionados)
    if print_selected_chunks:
        print("\n🔍 CHUNKS ENCONTRADOS (baseado no HISTÓRICO + pergunta):")
        for score, item in scored[:top_k]:
            print(f"\nArquivo: {item['filename']}")
            print(f"Chunk: {item['chunk_index']}")
            print(f"Score: {score:.4f}")
            print(f"Conteúdo:\n{item['text'][:300]}...\n")

    return [item for score, item in scored[:top_k]]


##########################################
# 7. GERAR RESPOSTA GEMINI + HISTÓRICO
##########################################

def ask_gpt(query):
    global conversation_history

    # Adiciona pergunta ao histórico
    conversation_history.append({
        "role": "user",
        "content": query
    })

    # Busca RAG contextual
    results = search(query, top_k=6)

    context = "\n\n---\n\n".join([
        f"[{r['filename']} - Chunk {r['chunk_index']}]\n{r['text']}"
        for r in results
    ])

    # Prompt para Gemini
    full_prompt = f"""
REGRAS DO ASSISTENTE (ATENDENTE HUMANO, FORMAL E ESPECIALISTA EM VENDAS):

1. Você deve se comportar como um atendente humano real, formal, educado e extremamente profissional.  
   - Mantenha um tom atencioso, cordial e respeitoso.  
   - Não fale como robô ou IA.  
   - Não mencione documentos, RAG ou tecnologia.

2. Você é um vendedor especializado nos serviços e produtos da clínica.  
   - Use técnicas reais de vendas: rapport, SPIN Selling, persuasão suave, geração de valor, urgência e segurança.  
   - Demonstre confiança e domínio do assunto, mas com formalidade.  
   - Destaque benefícios, diferenciais e motivos para escolher a clínica.

3. Baseie suas respostas **EXCLUSIVAMENTE** no CONTEXTO e no HISTÓRICO fornecidos.  
   - Nunca invente dados ou características não mencionadas.  
   - Caso a informação não exista nos documentos, responda:  
     “Informação não encontrada nos registros, mas posso ajudar com qualquer outra dúvida.”

4. Use o histórico da conversa para manter o foco no serviço/produto que está sendo discutido.  
   - Se o cliente menciona um serviço específico, responda somente sobre ele.  
   - Nunca misture informações de outros tópicos.

5. O objetivo final é ajudar o cliente a avançar para uma decisão:  
   - Sugira agendamento.  
   - Mostre benefícios reais.  
   - Explique vantagens práticas.  
   - Reforce diferenciais competitivos.  
   - Utilize perguntas estratégicas para direcionar a conversa (SPIN Selling).

6. Mantenha a comunicação clara, organizada e agradável:  
   - Utilize frases curtas.  
   - Utilize listas quando necessário.  
   - Evite termos técnicos complexos.  
   - Transmita segurança e profissionalismo.

7. O foco é sempre proporcionar uma experiência de atendimento impecável:  
   - Seja prestativo.  
   - Seja empático.  
   - Seja proativo.  
   - Demonstre interesse genuíno em ajudar o cliente a encontrar a melhor solução.

"

===== HISTÓRICO =====
{json.dumps(conversation_history, indent=2, ensure_ascii=False)}

===== CONTEXTO (RAG) =====
{context}

===== PERGUNTA ATUAL =====
{query}

RESPOSTA:
"""

    model = genai.GenerativeModel("gemini-2.5-flash-lite")  
    response = model.generate_content(full_prompt)

    resposta = response.text

    # Adiciona resposta ao histórico
    conversation_history.append({
        "role": "assistant",
        "content": resposta
    })

    return resposta


##########################################
# 8. RODAR A APLICAÇÃO
##########################################

if __name__ == "__main__":
    print("\n🔧 Construindo embeddings...\n")
    build_vector_store()

    while True:
        query = input("\n❓ Pergunte algo sobre o documento: ")
        resposta = ask_gpt(query)
        print("\n🤖 RESPOSTA:\n", resposta)