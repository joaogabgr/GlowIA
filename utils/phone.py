import re

def normalize_phone(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("+"):
        digits = digits[1:]
    return digits

def is_valid_phone(number: str) -> bool:
    digits = normalize_phone(number)
    return 10 <= len(digits) <= 15