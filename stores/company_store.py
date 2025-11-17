import os
import re
import json
import threading
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from bson import json_util
from collections import OrderedDict
from services.llm import embed_text
from services.tokenizer import chunk_text

class CompanyDataStore:
    def __init__(self, mongo_uri: str, local_dir: str, db_name: str, col_enterprise: str, max_cache_size: int = 12):
        self._cache = OrderedDict()
        self.max_cache = max_cache_size
        self._lock = threading.RLock()
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        if not mongo_uri:
            raise RuntimeError("MONGO_URI não definido")
        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[db_name]
        self.col_enterprise = self.db[col_enterprise]

    def _safe(self, s: str) -> str:
        return re.sub(r"[^a-z0-9_-]", "_", s.lower())

    def _embedding_file_for(self, name_or_id: str) -> str:
        return os.path.join(self.local_dir, f"{self._safe(name_or_id)}_embeddings.json")

    def _find_enterprise(self, company_key: str):
        key = company_key.strip()
        try:
            oid = ObjectId(key)
            doc = self.col_enterprise.find_one({"_id": oid})
            if doc:
                return doc
        except Exception:
            pass
        doc = self.col_enterprise.find_one({"empresa.nome": {"$regex": f"^{re.escape(key)}$", "$options": "i"}})
        if doc:
            return doc
        doc = self.col_enterprise.find_one({"empresa.cnpj": {"$regex": f"^{re.escape(key)}$", "$options": "i"}})
        if doc:
            return doc
        doc = self.col_enterprise.find_one({"empresa.contatos.site": {"$regex": f"^{re.escape(key)}$", "$options": "i"}})
        if doc:
            return doc
        raise FileNotFoundError("Empresa não encontrada")

    def _build_and_store_embeddings_local(self, enterprise_doc, id_str: str):
        text = json.dumps(enterprise_doc, ensure_ascii=False, default=json_util.default)
        chunks = chunk_text(text)
        vecstore = []
        for i, chunk in enumerate(chunks):
            vec = embed_text(chunk).tolist()
            vecstore.append({
                "filename": self._safe(id_str) + ".json",
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
        emb_fp = self._embedding_file_for(id_str)
        with open(emb_fp, "w", encoding="utf-8") as f:
            json.dump(vecstore, f, ensure_ascii=False, indent=2)
        return vecstore

    def get_vector_store(self, company_key: str):
        cache_key = company_key.lower()
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            enterprise_doc = self._find_enterprise(company_key)
            id_str = str(enterprise_doc.get("_id"))
            emb_fp = self._embedding_file_for(id_str)
            if os.path.exists(emb_fp):
                with open(emb_fp, "r", encoding="utf-8") as f:
                    store = json.load(f)
            else:
                store = self._build_and_store_embeddings_local(enterprise_doc, id_str)
            self._cache[cache_key] = store
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
            return store