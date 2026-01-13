import os
from pathlib import Path
from typing import Any, Dict, List
import argparse

from utils.image_utils import is_supported_image, infer_doc_type_from_filename
from src.extract import extract_document
from src.validate import validate_and_normalize
from src.io_utils import safe_stem, write_json
from src.decision import manual_decision_per_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def run() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="KYC Identity Verification PoV")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single document image to process",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(INPUT_DIR),
        help="Directory containing document images (default: data/input)",
    )

    args = parser.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("Missing FIREWORKS_API_KEY. Set it as an environment variable.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Resolve input files ---
    image_paths: List[Path] = []

    if args.file:
        p = Path(args.file)
        if not p.exists():
            raise RuntimeError(f"Input file not found: {p}")
        if not is_supported_image(str(p)):
            raise RuntimeError(f"Unsupported image type: {p}")
        image_paths = [p]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise RuntimeError(f"Input directory does not exist: {input_dir}")

        image_paths = [
            p for p in sorted(input_dir.iterdir())
            if p.is_file() and is_supported_image(str(p))
        ]

    if not image_paths:
        raise RuntimeError("No valid input documents found.")

    outputs: List[Dict[str, Any]] = []

    for path in image_paths:
        doc_type_hint = infer_doc_type_from_filename(path.name)

        raw = extract_document(str(path), doc_type_hint=doc_type_hint)
        validated = validate_and_normalize(raw)

        validated.pop("manual_decision", None)
        validated.pop("decision", None)

        decision_obj = manual_decision_per_document(validated)

        payload = {
            "decision": decision_obj["decision"],
            "file_name": path.name,
            "doc_type": decision_obj.get("doc_type", validated.get("doc_type")),
            "risk_level": decision_obj["risk_level"],
            "decision_reasons": decision_obj["decision_reasons"],
            "required_fields_missing": decision_obj["required_fields_missing"],
            "confidence_used": decision_obj["confidence_used"],
            "doc_type_hint": doc_type_hint,
            "extracted": validated,
        }

        out_name = f"{safe_stem(path.name)}.json"
        out_path = OUTPUT_DIR / out_name
        write_json(out_path, payload)

        outputs.append({
            "file_name": path.name,
            "decision": decision_obj["decision"],
            "risk_level": decision_obj["risk_level"],
            "confidence_used": decision_obj["confidence_used"],
            "output_file": str(out_path),
        })

    return {"outputs": outputs}



if __name__ == "__main__":
    res = run()
    print("Per-document outputs:")
    for d in res["outputs"]:
        print(
            f" - {d['file_name']}: {d['decision']} "
            f"(risk={d['risk_level']}, conf={d['confidence_used']:.2f}) -> {d['output_file']}"
        )