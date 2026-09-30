"""
Batch processor: document-level + segment-level routing -> single or hybrid chunking plan.
"""
import hashlib
import json
from pathlib import Path

from pdf_parser import profile_full_document
from laya_classifier import DocumentClassifier
from strategies import (CONFIG as C, STRATEGY_PRIORITY, ALLOWED_BY_LABEL,
                        LOSS_CRITICAL, SOFT_STRATEGIES)

INPUT_DIR = Path("input_docs")
OUTPUT_DIR = Path("output")


def setup_folders():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def resolve_fallback(h: dict, label: str | None) -> tuple[str, str]:
    """Deterministic priority guardrails. Respects the segment label's allowed set."""
    if h["is_suspected_scan"] or (not h["is_digitally_native"] and h["total_images"] > 0):
        pick, why = "OCR_Pipeline", "suspected scan / non-native text"
    elif h["has_high_visual_density"]:
        pick, why = "Multimodal_Layout_Aware", "high visual density"
    elif h["total_tables"] >= 2 or h["table_coverage"] >= C["table_area_ratio"]:
        pick, why = "Document_Based", "table dominance"
    elif h["technical_ratio"] >= C["technical_symbol_ratio"] and h["avg_chars_per_page"] >= C["technical_min_chars"]:
        pick, why = "Semantic", "dense technical text"
    else:
        pick, why = "Recursive", "standard prose"
    allowed = ALLOWED_BY_LABEL.get(label)
    if allowed and pick not in allowed:
        pick = sorted(allowed, key=STRATEGY_PRIORITY.index)[0]
        why += f" (constrained to {label} options)"
    return pick, f"Guardrail: {why}"


def decide(clf: DocumentClassifier, h, sample, n_pages, label=None) -> dict:
    r = clf.classify(h, sample, n_pages, label)
    if r["routing_source"] == "Deterministic Pruning":
        return r
    low_conf = r["recommended_strategy"] is None or r["confidence_score"] < C["laya_min_confidence"]
    if low_conf:
        strat, why = resolve_fallback(h, label)
        r["ai_raw_choice"], r["ai_raw_confidence"] = r["recommended_strategy"], r["confidence_score"]
        r["recommended_strategy"], r["confidence_score"] = strat, None   # no fake 1.0
        r["routing_source"], r["routing_note"] = "Deterministic Guardrail", why
    else:
        top = sorted(r["probabilities"].values(), reverse=True)[:2]
        r["needs_review"] = len(top) == 2 and (top[0] - top[1]) < C["review_margin"]
    return r


def _pages(s, e):
    return str(s) if s == e else f"{s}-{e}"


def absorb_small_soft(segs: list[dict]) -> list[dict]:
    """Tiny Semantic/Recursive segments are not worth a separate pipeline."""
    weight = {}
    for s in segs:
        if s["strategy"] in SOFT_STRATEGIES:
            weight[s["strategy"]] = weight.get(s["strategy"], 0) + s["n_pages"]
    if not weight:
        return segs
    target = max(sorted(weight), key=lambda k: weight[k])
    for s in segs:
        if (s["strategy"] in SOFT_STRATEGIES and s["strategy"] != target
                and s["n_pages"] < C["min_soft_segment_pages"]):
            s["absorbed_from"], s["strategy"] = s["strategy"], target
    return segs


def merge_adjacent(segs: list[dict]) -> list[dict]:
    out = []
    for s in segs:
        if out and out[-1]["strategy"] == s["strategy"] and out[-1]["end_page"] + 1 == s["start_page"]:
            out[-1]["end_page"], out[-1]["n_pages"] = s["end_page"], out[-1]["n_pages"] + s["n_pages"]
            out[-1]["routing_source"] = "merged"
        else:
            out.append(dict(s))
    return out


def build_plan(profile: dict, clf: DocumentClassifier) -> dict:
    segs = []
    for sg in profile["segments"]:
        r = decide(clf, sg["heuristics"], sg["text_sample"], sg["n_pages"], sg["dominant_label"])
        segs.append({"start_page": sg["start_page"], "end_page": sg["end_page"], "n_pages": sg["n_pages"],
                     "layout_label": sg["dominant_label"], "strategy": r["recommended_strategy"],
                     "routing_source": r["routing_source"], "confidence_score": r["confidence_score"],
                     "probabilities": r["probabilities"], "needs_review": r.get("needs_review", False),
                     "options": r.get("options", []),
                     **{k: r[k] for k in ("ai_raw_choice", "ai_raw_confidence", "routing_note") if k in r}})

    if len(segs) == 1:
        doc_cls = {"recommended_strategy": segs[0]["strategy"], "note": "single homogeneous segment"}
    else:
        doc_cls = decide(clf, profile["heuristics"], profile["text_sample"], profile["total_pages"], None)

    segs = merge_adjacent(absorb_small_soft(segs))
    strategies = sorted({s["strategy"] for s in segs}, key=STRATEGY_PRIORITY.index)
    mode = "single" if len(strategies) == 1 else "hybrid"

    routes = {st: [_pages(s["start_page"], s["end_page"]) for s in segs if s["strategy"] == st]
              for st in strategies}
    return {
        "mode": mode,
        "primary_strategy": max(strategies, key=lambda st: sum(s["n_pages"] for s in segs if s["strategy"] == st)),
        "loss_critical_present": sorted(set(strategies) & LOSS_CRITICAL),
        "routes": routes,
        "segments": segs,
        "document_level_classification": {k: v for k, v in doc_cls.items() if k != "options"},
        "needs_review": any(s["needs_review"] for s in segs),
    }


def analyze_pdf(pdf_path: Path, clf: DocumentClassifier):
    """Profile + plan one PDF and return the result dict (shared by CLI and Streamlit UI)."""
    profile = profile_full_document(str(pdf_path))
    if not profile:
        return None
    plan = build_plan(profile, clf)
    return {
        "file": pdf_path.name,
        "file_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "document_stats": profile["heuristics"],
        "page_labels": profile["page_labels"],
        "page_errors": profile["page_errors"],
        "plan": plan,
    }


def save_result(final: dict, out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{Path(final['file']).stem}_strategy.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True))
    return out


def process_all_pdfs():
    print("Loading Laya model into memory...")
    clf = DocumentClassifier(cache_path=str(OUTPUT_DIR / ".laya_cache.json"))

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"), key=lambda p: p.name)
    if not pdf_files:
        print(f"\n[!] No PDFs found in '{INPUT_DIR}'. Drop PDF files there and re-run.")
        return

    for pdf_path in pdf_files:
        print(f"\nProfiling: {pdf_path.name}...")
        try:
            final = analyze_pdf(pdf_path, clf)
        except Exception as e:                           # encrypted / corrupt
            print(f" -> ERROR {type(e).__name__}")
            continue
        if not final:
            continue
        save_result(final)
        plan = final["plan"]
        print(f" -> {plan['mode'].upper()}: {plan['routes']}"
              + ("  [REVIEW]" if plan["needs_review"] else ""))

    clf.save_cache()


if __name__ == "__main__":
    setup_folders()
    process_all_pdfs()