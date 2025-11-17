import time
import numpy as np
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, MODEL_NAME

genai.configure(api_key=GEMINI_API_KEY)

def embed_text(text: str):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query",
        request_options={"timeout": 10}
    )
    return np.array(result["embedding"], dtype=np.float32)

def generate_answer(prompt: str, retries: int = 3):
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