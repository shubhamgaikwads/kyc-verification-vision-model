from typing import Any, Dict, List

def manual_decision_per_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    file_name = doc.get("_meta", {}).get("source_file", "unknown_file")
    doc_type = doc.get("doc_type", "unknown")
    warnings: List[str] = doc.get("warnings", []) or []
    conf = float(doc.get("final_confidence", doc.get("extraction_confidence", 0.0)) or 0.0)

    required = ["full_name", "date_of_birth", "document_number"]
    if doc_type in ("passport", "drivers_license"):
        required.append("date_of_expiry")

    missing = [k for k in required if not doc.get(k)]

    hard_reject_signals = [
        "Document appears expired.",
        "MRZ passport number check digit failed",
        "MRZ DOB check digit failed",
        "MRZ expiry check digit failed",
        "DOB appears to be in the future.",
        "Implausible DOB:",                 # from plausibility checks
        "Implausible DOB year",
        "Expiry date is on/before issue date",
    ]
    hard_reject = any(any(sig in w for sig in hard_reject_signals) for w in warnings)

    reasons: List[str] = []
    if missing:
        reasons.append(f"Missing required fields: {', '.join(missing)}")
    if hard_reject:
        reasons.append("Hard fail: validation/plausibility issue detected.")
    if conf < 0.60:
        reasons.append(f"Low extraction confidence ({conf:.2f}).")

    # Decide in pass/review/fail
    if hard_reject or len(missing) >= 2 or conf < 0.45:
        decision = "fail"
        risk_level = "high"
    elif missing or conf < 0.75 or len(warnings) > 0:
        decision = "review"
        risk_level = "medium"
    else:
        decision = "pass"
        risk_level = "low"

    return {
        "file_name": file_name,
        "doc_type": doc_type,
        "decision": decision,                     # <-- KEY YOUR main.py EXPECTS
        "risk_level": risk_level,
        "decision_reasons": reasons if reasons else ["Meets thresholds for this document."],
        "required_fields_missing": missing,
        "confidence_used": conf,
    }
