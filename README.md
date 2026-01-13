This repository demonstrates an end-to-end Proof of Value (PoV) for Identity Verification (IDV) in an FSI KYC workflow using Fireworks AI Vision-Language Models.

The system extracts identity attributes from documents (passports, driver’s licenses), validates plausibility, and produces a per-document decision: pass, review, or fail.

🔹 Architecture

Image → Fireworks VLM → Structured Extraction → Validation →
Plausibility Checks → Decision Engine

🔹 Key Design Decisions & Tradeoffs
Decision	                  Why	                                        Tradeoff
Vision-Language Model	      Handles varied layouts & low-quality scans	Higher cost vs OCR
Strict JSON schema	          Prevents hallucination & enforces structure	Requires defensive parsing
Single-document evaluation	  Mirrors real KYC workflows	                No cross-doc linking
Conservative decisioning	  FSI compliance friendly	                    More “review” cases
No inference on missing data  Avoids false positives	                    Lower automation rate

🔹 Input
Accepts PNG/JPG documents
Supports passports and driver’s licenses
One or many documents can be processed independently

🔹 Output (Per Document)
Each document produces a standalone JSON with:
Decision at the top (pass / review / fail)
Risk level
Explanation
Extracted fields
Confidence scores
Warnings
Example:
{
  "decision": "review",
  "risk_level": "medium",
  "decision_reasons": [
    "Document expired",
    "Date ambiguity detected"
  ],
  "extracted": { ... }
}

#🔹 How to Run
1. Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Set Fireworks API key
export FIREWORKS_API_KEY=your_key_here
3. Add documents
cp *.png data/input/
4. Run
python -m src.main



#🔹 Limitations & Future Improvements

Cross-document identity matching
Face matching
Country-specific date formats
Retry & ensemble extraction
Production-grade audit logging
