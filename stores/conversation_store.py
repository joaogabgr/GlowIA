import os
import json
import re
import threading

class ConversationStore:
    MAX_HISTORY = 30

    def __init__(self, base_dir: str):
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