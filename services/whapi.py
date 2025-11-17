import json
import logging
from typing import Optional
from urllib import request, error
from config.settings import WHAPI_BASE_URL, WHAPI_TOKEN, WHAPI_TIMEOUT
from utils.phone import normalize_phone, is_valid_phone

def send_text_message(to: str, body: str, timeout: Optional[int] = None, logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger("whapi")

    if not WHAPI_TOKEN:
        raise RuntimeError("WHAPI_TOKEN não configurado")

    normalized = normalize_phone(to)
    if not is_valid_phone(normalized):
        raise ValueError("Número de telefone inválido")

    url = WHAPI_BASE_URL.rstrip("/") + "/messages/text"
    payload = {"to": normalized, "body": body}
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {WHAPI_TOKEN}",
        "Content-Type": "application/json",
    }

    req = request.Request(url, data=data, headers=headers, method="POST")
    req_timeout = timeout if timeout is not None else WHAPI_TIMEOUT

    logger.info(f"Enviando mensagem para {normalized}")
    logger.debug(f"Payload: {payload}")

    try:
        with request.urlopen(req, timeout=req_timeout) as resp:
            resp_text = resp.read().decode("utf-8")
            logger.info(f"Mensagem enviada. Status {resp.status}")
            logger.debug(f"Resposta: {resp_text}")
            try:
                return json.loads(resp_text)
            except Exception:
                return {"status": resp.status, "raw": resp_text}
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8") if hasattr(e, "read") else ""
        logger.error(f"Falha HTTP {e.code}: {body_text}")
        raise
    except error.URLError as e:
        logger.error(f"Erro de conexão: {e}")
        raise