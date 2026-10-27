import re


_MEDICAL_ADVICE_PATTERNS = (
    r"\b(?:what|which)\b.{0,80}\b"
    r"(?:medications?|drugs?|antibiotics?|treatments?|doses?|dosages?)\b"
    r".{0,80}\b(?:should|could|can|would)\b.{0,40}\b"
    r"(?:i|we|you|(?:the )?patient)\b.{0,30}\b"
    r"(?:start|stop|increase|decrease|change|adjust|switch|prescribe|take|use|give|administer)\b",
    r"\b(?:should|could|can|would)\s+(?:i|we|you|(?:the )?patient)\s+"
    r"(?:start|stop|increase|decrease|change|adjust|switch|prescribe|take|use|receive)\b"
    r".{0,60}\b(?:medications?|drugs?|doses?|dosages?|antibiotics?|treatment)\b",
    r"\b(?:what|which)\s+(?:medications?|drugs?|doses?|dosages?|antibiotics?|treatment)\s+"
    r"(?:should|could|can|would)\b",
    r"\b(?:recommend|suggest|prescribe)\b.{0,80}\b"
    r"(?:medications?|drugs?|doses?|dosages?|antibiotics?|treatment)\b",
    r"\b(?:best treatment|diagnose(?: me| the patient| my symptoms)?|"
    r"what is wrong with me|what condition do i have)\b",
)

_COMPILED_ADVICE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _MEDICAL_ADVICE_PATTERNS
)


def is_medical_advice_request(question: str) -> bool:
    return any(pattern.search(question) for pattern in _COMPILED_ADVICE_PATTERNS)


def get_retrieval_mode(question: str) -> str:
    normalized = question.casefold()

    if "admission" in normalized or "admitted" in normalized:
        return "admission"
    if any(term in normalized for term in ("medication", "medicine", "drug", "dosage", "dose")):
        return "medications"
    if any(term in normalized for term in ("report", "summar", "recent", "latest", "last few")):
        return "reports"
    return "semantic"
