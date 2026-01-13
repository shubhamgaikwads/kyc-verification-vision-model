from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _dob_match(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    # ISO dates recommended; if not ISO, treat as unknown
    if not a or not b:
        return None
    # Strict match only for PoV; you can add fuzzy date parsing if needed
    return a == b


def _choose_best(docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not docs:
        return None
    return sorted(docs, key=lambda d: float(d.get("final_confidence", 0.0)), reverse=True)[0]


def cross_document_check(all_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    passports = [d for d in all_docs if d.get("doc_type") == "passport"]
    dls = [d for d in all_docs if d.get("doc_type") == "drivers_license"]

    best_passport = _choose_best(passports)
    best_dl = _choose_best(dls)

    result: Dict[str, Any] = {
        "passport_used": best_passport.get("_meta", {}).get("source_file") if best_passport else None,
        "drivers_license_used": best_dl.get("_meta", {}).get("source_file") if best_dl else None,
        "checks": [],
        "decision": "review",
        "reasons": [],
    }

    if not best_passport or not best_dl:
        result["decision"] = "review"
        result["reasons"].append("Missing passport or driver's license for cross-check.")
        return result

    p_name = best_passport.get("full_name")
    d_name = best_dl.get("full_name")
    p_dob = best_passport.get("date_of_birth")
    d_dob = best_dl.get("date_of_birth")

    name_sim = _name_similarity(p_name, d_name)
    dob_match = _dob_match(p_dob, d_dob)

    result["checks"].append({"type": "name_similarity", "value": round(name_sim, 3)})
    result["checks"].append({"type": "dob_match", "value": dob_match})

    # Decision logic (PoV)
    # - PASS: strong name similarity + dob match + both docs reasonably confident
    # - FAIL: strong contradiction (low similarity + dob mismatch)
    # - REVIEW: everything else
    p_conf = float(best_passport.get("final_confidence", 0.0))
    d_conf = float(best_dl.get("final_confidence", 0.0))

    if name_sim >= 0.90 and dob_match is True and p_conf >= 0.70 and d_conf >= 0.70:
        result["decision"] = "pass"
    elif name_sim < 0.75 and dob_match is False:
        result["decision"] = "fail"
        result["reasons"].append(f"Low name similarity ({name_sim:.2f}) and DOB mismatch.")
    else:
        result["decision"] = "review"
        if name_sim < 0.90:
            result["reasons"].append(f"Name similarity below threshold ({name_sim:.2f}).")
        if dob_match is not True:
            result["reasons"].append("DOB mismatch or missing.")
        if p_conf < 0.70 or d_conf < 0.70:
            result["reasons"].append("Low extraction confidence on one or more documents.")

    return result
