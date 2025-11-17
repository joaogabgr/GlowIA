import re

def sanitize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\x00-\x1F]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def format_for_whatsapp(text: str) -> str:
    if text is None:
        return ""
    s = text.replace("\r\n", "\n")
    s = s.replace("\\n", "\n")
    s = re.sub(r"\*\*(.*?)\*\*", r"*\1*", s)
    return s