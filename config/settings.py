import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL") or "mongodb://localhost:27017"
MONGO_DB = os.getenv("MONGO_DB", "GlowAI")
MONGO_COLLECTION_ENTERPRISE = os.getenv("MONGO_COLLECTION_ENTERPRISE", "entherprise")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "data/empresas")
CONVERSATIONS_DIR = "data/conversations"

with open("persona.txt", "r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()

WHAPI_BASE_URL = os.getenv("WHAPI_BASE_URL", "https://gate.whapi.cloud/")
WHAPI_TOKEN = os.getenv("WHAPI_TOKEN")
WHAPI_TIMEOUT = int(os.getenv("WHAPI_TIMEOUT", "30"))