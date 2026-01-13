import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from utils.image_utils import image_to_data_uri


DEFAULT_VLM_MODEL = os.environ.get(
    "FIREWORKS_VLM_MODEL",
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
)

FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1",
)



# Robust JSON parsing (PoV-grade)
def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON from model output defensively.

    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Unterminated JSON object in model output.")


def _get_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY environment variable.")

    return OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)



# Fireworks JSON schema
EXTRACTION_SCHEMA: Dict[str, Any] = {
    "name": "kyc_document_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "doc_type": {"type": "string", "enum": ["passport", "drivers_license", "unknown"]},
            "issuing_country_or_state": {"type": ["string", "null"]},
            "document_number": {"type": ["string", "null"]},
            "full_name": {"type": ["string", "null"]},
            "given_name": {"type": ["string", "null"]},
            "surname": {"type": ["string", "null"]},
            "date_of_birth": {"type": ["string", "null"], "description": "Prefer ISO YYYY-MM-DD when possible"},
            "date_of_issue": {"type": ["string", "null"]},
            "date_of_expiry": {"type": ["string", "null"]},
            "nationality": {"type": ["string", "null"]},
            "sex": {"type": ["string", "null"]},
            "address": {"type": ["string", "null"]},
            "mrz": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    # NOTE: raw can contain newlines and < characters; keep it short.
                    "raw": {"type": ["string", "null"]},
                    "line1": {"type": ["string", "null"]},
                    "line2": {"type": ["string", "null"]},
                },
                "required": ["raw", "line1", "line2"],
            },
            "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "field_confidences": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "evidence": {
                "type": "object",
                "description": "Short snippets supporting key fields (no full document dump).",
                "additionalProperties": {"type": "string"},
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "doc_type",
            "issuing_country_or_state",
            "document_number",
            "full_name",
            "given_name",
            "surname",
            "date_of_birth",
            "date_of_issue",
            "date_of_expiry",
            "nationality",
            "sex",
            "address",
            "mrz",
            "extraction_confidence",
            "field_confidences",
            "evidence",
            "warnings",
        ],
    },
}


def _build_prompts(doc_type_hint: str) -> Dict[str, str]:
    """
    Prompt updates to improve "correctly get document data":
    - Force one-line strings (prevents broken JSON due to newlines)
    - Keep evidence short and avoid quotes (prevents unterminated strings)
    - Prefer MRZ line1/line2 for passports (more reliable than free-form text)
    - Explicitly ban commentary outside JSON
    """
    system = (
        "You are a KYC identity document extraction engine.\n"
        "Extract ONLY what is clearly visible and legible in the image.\n"
        "If a field is missing or unclear, set it to null.\n"
        "Do NOT guess. Do NOT infer.\n"
        "Return ONLY valid JSON that matches the provided schema.\n"
        "All string values MUST be single-line (no newline characters).\n"
        "Do not include double quotes inside string values.\n"
        "Keep evidence snippets short (<= 60 characters).\n"
        f"Document type hint: {doc_type_hint}\n"
    )

    user = (
        "Extract identity fields needed for KYC identity verification from this image.\n"
        "Rules:\n"
        "- Output must follow the JSON schema exactly.\n"
        "- Prefer ISO date format YYYY-MM-DD when confident; else return the raw date string.\n"
        "- If multiple candidates exist for a field, choose the most likely and add a warning.\n"
        "- For passports: prioritize MRZ line1/line2 if present.\n"
        "- Evidence must be SHORT snippets for: full_name, date_of_birth, document_number, date_of_expiry.\n"
        "- Never output full document text.\n"
        "- Return JSON only. No extra text.\n"
    )

    return {"system": system, "user": user}



# Safe fallback payload
def _fallback_payload() -> Dict[str, Any]:
    return {
        "doc_type": "unknown",
        "issuing_country_or_state": None,
        "document_number": None,
        "full_name": None,
        "given_name": None,
        "surname": None,
        "date_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "nationality": None,
        "sex": None,
        "address": None,
        "mrz": None,
        "extraction_confidence": 0.0,
        "field_confidences": {},
        "evidence": {},
        "warnings": [
            "Model output could not be parsed as valid JSON.",
            "Extraction incomplete; manual review required.",
        ],
    }


def extract_document(
    image_path: str,
    doc_type_hint: str = "unknown",
    model: Optional[str] = None,
    max_side: int = 1600,
    max_tokens: int = 800,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Calls Fireworks VLM to extract structured fields from an identity document image.

    Reliability updates:
    - Lower max_tokens and temperature to reduce malformed JSON.
    - Safer JSON extraction (brace counting).
    - Deterministic fallback object (forces review/fail downstream).
    """
    client = _get_client()
    model = model or DEFAULT_VLM_MODEL

    prompts = _build_prompts(doc_type_hint)
    data_uri = image_to_data_uri(image_path, max_side=max_side)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts["user"]},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content or ""

    try:
        out = safe_json_loads(content)
    except Exception:
        out = _fallback_payload()

    # Attach metadata (safe—no raw image)
    out["_meta"] = {
        "source_file": os.path.basename(image_path),
        "model": model,
        "doc_type_hint": doc_type_hint,
    }
    return out
