"""
Laya router with (1) dynamic pruning, (2) label-constrained option sets,
(3) short-circuit when only one option is left, (4) input-keyed cache so identical
input always returns the identical Laya answer (reproducibility + speed).
"""
import hashlib
import json
from pathlib import Path

import laya
from strategies import CHUNKING_REGISTRY, ALLOWED_BY_LABEL, STRATEGY_PRIORITY


class DocumentClassifier:
    def __init__(self, cache_path: str | None = None):
        self.agent = laya.Router()
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def save_cache(self):
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True))

    def _prune(self, h: dict, label: str | None) -> dict:
        valid = dict(CHUNKING_REGISTRY)
        if h["is_digitally_native"] and not h["is_suspected_scan"]:
            valid.pop("OCR_Pipeline", None)
        if h["total_images"] == 0:
            valid.pop("Multimodal_Layout_Aware", None)
        if h["total_tables"] == 0:
            valid.pop("Document_Based", None)
        if label in ALLOWED_BY_LABEL:                    # segment scope: Laya only arbitrates real ambiguity
            allowed = set(ALLOWED_BY_LABEL[label])
            narrowed = {k: v for k, v in valid.items() if k in allowed}
            valid = narrowed or {k: v for k, v in CHUNKING_REGISTRY.items() if k in allowed}
        return {k: valid[k] for k in STRATEGY_PRIORITY if k in valid}   # fixed order

    def classify(self, h: dict, text_sample: str, n_pages: int, label: str | None = None) -> dict:
        valid = self._prune(h, label)
        if len(valid) == 1:
            only = next(iter(valid))
            return {"recommended_strategy": only, "confidence_score": None, "probabilities": {},
                    "options": [only], "routing_source": "Deterministic Pruning"}

        state = (
            f"=== SCOPE ===\n- Pages in scope: {n_pages}\n- Layout label: {label or 'MIXED/WHOLE DOCUMENT'}\n\n"
            f"=== METRICS ===\n"
            f"- Significant images: {h['total_images']} ({h['images_per_page']}/page, avg page coverage {h['image_coverage']})\n"
            f"- Tables: {h['total_tables']} (avg page coverage {h['table_coverage']})\n"
            f"- Avg chars/page: {h['avg_chars_per_page']}\n"
            f"- Math/citation symbol ratio: {h['technical_ratio']}\n"
            f"- Digitally native: {h['is_digitally_native']}\n"
            f"- Scan fraction: {h['scan_fraction']}\n"
            f"- Figure-page fraction: {h['figure_fraction']}\n"
            f"- Table-page fraction: {h['table_fraction']}\n\n"
            f"=== CONTENT SAMPLE ===\n{text_sample[:1500]}"
        )
        key = hashlib.sha256((state + "|" + ",".join(valid)).encode()).hexdigest()
        if key in self.cache:
            out = dict(self.cache[key])
            out["routing_source"] = "Laya AI Model (cached)"
            return out

        try:
            resp = self.agent.predict(
                state=state,
                questions={"overall_strategy": {
                    "type": "choice",
                    "instructions": "Select the single best chunking strategy for the content in scope.",
                    "criteria": valid,
                }},
            )
            ans = resp["answers"]["overall_strategy"]
            probs = {k: round(float(v), 4) for k, v in sorted(ans.get("probabilities", {}).items())}
            out = {"recommended_strategy": ans["choice"],
                   "confidence_score": round(float(ans["answer_confidence"]), 4),
                   "probabilities": probs, "options": list(valid)}
        except Exception as e:                           # model failure -> deterministic fallback in main
            return {"recommended_strategy": None, "confidence_score": 0.0, "probabilities": {},
                    "options": list(valid), "routing_source": f"Laya error: {type(e).__name__}"}

        self.cache[key] = out
        return {**out, "routing_source": "Laya AI Model"}