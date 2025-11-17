import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, max_tokens: int = 300):
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = ENCODING.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks