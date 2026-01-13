from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

from dateutil import parser as dateparser
import re
from dateutil import parser as dateparser

def _parse_us_date_from_evidence(s: str, dayfirst: bool) -> str | None:
    # expects like 01/08/2026
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", s)
    if not m:
        return None
    try:
        dt = dateparser.parse(m.group(1), dayfirst=dayfirst, yearfirst=False)
        return dt.date().isoformat()
    except Exception:
        return None


def _safe_iso_date(
    value: Optional[str],
    *,
    dayfirst: bool
) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = dateparser.parse(
            value,
            dayfirst=dayfirst,
            yearfirst=False
        )
        return dt.date().isoformat()
    except Exception:
        return None


def _normalize_name(value: Optional[str]) -> Optional[str]:
    """Uppercase, collapse whitespace, replace commas with spaces."""
    if not value:
        return None
    cleaned = " ".join(value.replace(",", " ").split())
    return cleaned.upper() if cleaned else None


def _normalize_doc_number(value: Optional[str]) -> Optional[str]:
    """Keep only alphanumerics, uppercase."""
    if not value:
        return None
    cleaned = "".join(ch for ch in value if ch.isalnum())
    return cleaned.upper() if cleaned else None



# MRZ checksum helpers (basic, PoV-grade)
MRZ_CHAR_VALUES = {str(i): i for i in range(10)}
MRZ_CHAR_VALUES.update({chr(ord("A") + i): 10 + i for i in range(26)})
MRZ_CHAR_VALUES["<"] = 0
MRZ_WEIGHTS = [7, 3, 1]


def _mrz_value(ch: str) -> int:
    return MRZ_CHAR_VALUES.get(ch, 0)


def _mrz_checkdigit(field: str) -> str:
    total = 0
    for i, ch in enumerate(field):
        total += _mrz_value(ch) * MRZ_WEIGHTS[i % 3]
    return str(total % 10)


def _extract_mrz_lines(mrz_raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to return (line1, line2). Supports multiline and 88-char concatenated MRZ."""
    if not mrz_raw:
        return None, None

    mrz_raw = mrz_raw.strip().replace(" ", "")
    parts = [p.strip() for p in mrz_raw.splitlines() if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]

    # Sometimes MRZ is returned as one long string; split into two 44-char lines (TD3).
    if len(mrz_raw) >= 88:
        return mrz_raw[:44], mrz_raw[44:88]

    return None, None


def _validate_passport_mrz(line2: str) -> Optional[str]:
    """
    Very basic passport MRZ (TD3) line2 checks:
      - passport number check digit
      - DOB check digit
      - expiry check digit
    """
    if not line2 or len(line2) < 44:
        return None

    passport_number = line2[0:9]
    passport_cd = line2[9]

    dob = line2[13:19]
    dob_cd = line2[19]

    expiry = line2[21:27]
    expiry_cd = line2[27]

    if _mrz_checkdigit(passport_number) != passport_cd:
        return "MRZ passport number check digit failed"
    if _mrz_checkdigit(dob) != dob_cd:
        return "MRZ DOB check digit failed"
    if _mrz_checkdigit(expiry) != expiry_cd:
        return "MRZ expiry check digit failed"

    return None


def validate_and_normalize(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extracted fields and add validation / plausibility warnings.
    Also computes a 'final_confidence' derived from extraction_confidence with penalties.
    """
    out = dict(extracted)
    out.setdefault("warnings", [])
    out.setdefault("field_confidences", {})
    out.setdefault("evidence", {})

    # --- Normalize core fields ---
    out["full_name"] = _normalize_name(out.get("full_name"))
    out["given_name"] = _normalize_name(out.get("given_name"))
    out["surname"] = _normalize_name(out.get("surname"))
    out["document_number"] = _normalize_doc_number(out.get("document_number"))


    evidence = out.get("evidence") or {}
    issuing = (out.get("issuing_country_or_state") or "").lower()

    dayfirst = False
    if "india" in issuing:
        dayfirst = True

    # check expiry evidence vs normalized expiry
    ev_exp = evidence.get("date_of_expiry")
    norm_exp = out.get("date_of_expiry")

    if ev_exp and norm_exp:
        ev_exp_iso = _parse_us_date_from_evidence(ev_exp, dayfirst=dayfirst)
        if ev_exp_iso and ev_exp_iso != norm_exp:
            out["warnings"].append(
                f"Date mismatch: evidence expiry '{ev_exp}' -> {ev_exp_iso} but normalized date_of_expiry is {norm_exp}."
            )


    # --- Normalize dates ---
    dob_iso = _safe_iso_date(out.get("date_of_birth"), dayfirst=dayfirst)
    doi_iso = _safe_iso_date(out.get("date_of_issue"), dayfirst=dayfirst)
    doe_iso = _safe_iso_date(out.get("date_of_expiry"), dayfirst=dayfirst)

    if dob_iso:
        out["date_of_birth"] = dob_iso
    if doi_iso:
        out["date_of_issue"] = doi_iso
    if doe_iso:
        out["date_of_expiry"] = doe_iso

    today = date.today()

    # --- Basic sanity checks ---
    if dob_iso:
        dob = date.fromisoformat(dob_iso)
        if dob > today:
            out["warnings"].append("DOB appears to be in the future.")
        # Keep the original check but tighten downstream with plausibility rules
        if (today.year - dob.year) > 120:
            out["warnings"].append("DOB implies age > 120; verify extraction.")

    if doe_iso:
        exp = date.fromisoformat(doe_iso)
        if exp < today:
            out["warnings"].append("Document appears expired.")

    # --- Plausibility checks (KYC sanity: catch 'Benjamin Franklin' / 1706 etc.) ---
    if dob_iso:
        dob = date.fromisoformat(dob_iso)
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        if age < 0:
            out["warnings"].append("DOB is in the future (implausible).")
        if age > 115:
            out["warnings"].append(
                f"Implausible DOB: age={age} (>115) for modern identity documents."
            )
        if dob.year < 1900:
            out["warnings"].append(
                f"Implausible DOB year ({dob.year}) for modern identity documents."
            )

    if doi_iso and doe_iso:
        doi = date.fromisoformat(doi_iso)
        doe = date.fromisoformat(doe_iso)

        if doe <= doi:
            out["warnings"].append("Expiry date is on/before issue date (implausible).")

        if (doe.year - doi.year) > 20:
            out["warnings"].append(
                "Unusually long validity period (>20 years); verify extraction."
            )

    # --- MRZ validation (passport only) ---
    if out.get("doc_type") == "passport":
        mrz_obj = out.get("mrz") or {}
        line1 = mrz_obj.get("line1")
        line2 = mrz_obj.get("line2")
        raw = mrz_obj.get("raw")

        # If model provided only raw, try to split into lines
        if (not line1 or not line2) and raw:
            l1, l2 = _extract_mrz_lines(raw)
            if not line1:
                mrz_obj["line1"] = l1
            if not line2:
                mrz_obj["line2"] = l2
            out["mrz"] = mrz_obj

        if mrz_obj.get("line2"):
            mrz_issue = _validate_passport_mrz(mrz_obj["line2"])
            if mrz_issue:
                out["warnings"].append(mrz_issue)

    # --- Confidence adjustment (PoV heuristic) ---
    base = float(out.get("extraction_confidence", 0.5))
    penalty = 0.0

    # Penalize missing critical fields
    critical_fields = ("full_name", "date_of_birth", "document_number")
    missing_critical = [k for k in critical_fields if not out.get(k)]
    if missing_critical:
        out["warnings"].append(f"Missing critical fields: {', '.join(missing_critical)}")
        penalty += 0.15

    # Penalize per warning (bounded)
    penalty += min(0.25, 0.05 * max(0, len(out["warnings"]) - 1))

    out["final_confidence"] = max(0.0, min(1.0, base - penalty))
    return out
