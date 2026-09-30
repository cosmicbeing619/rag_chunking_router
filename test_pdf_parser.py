"""
pdfplumber layout extraction: per-page profiling -> page labels -> segments.
Deterministic: pure function of PDF content + CONFIG.
"""
import math
import re
from collections import Counter
import pdfplumber
from strategies import CONFIG as C



_TECH = re.compile(r"[∑∫∂∇≤≥≈≠±×÷∈∉⊂⊆∪∩→⇒∀∃αβγδεζηθλμπσφψω]|\[\d+\]")
_TEXT_LABELS = {"PROSE", "TECHNICAL"}


def _profile_page(page) -> dict:
    w, h = float(page.width), float(page.height)
    area = max(w * h, 1.0)
    text = (page.extract_text() or "").strip()          # no layout=True: avoids whitespace padding
    chars = len(re.sub(r"\s+", "", text))

    img_cov, sig_imgs, biggest = 0.0, 0, 0.0
    for im in page.images:
        iw = max(0.0, min(float(im["x1"]), w) - max(float(im["x0"]), 0.0))
        ih = max(0.0, min(float(im["bottom"]), h) - max(float(im["top"]), 0.0))
        r = iw * ih / area
        if r >= C["min_image_area_ratio"]:              # filters logos/icons/rules
            sig_imgs += 1
            img_cov += r
            biggest = max(biggest, r)
    img_cov = min(img_cov, 1.0)

    tbl_cov, tbl_n = 0.0, 0
    for t in page.find_tables():
        rows = t.rows
        cols = len(rows[0].cells) if rows else 0
        if len(rows) >= C["table_min_rows"] and cols >= C["table_min_cols"]:
            x0, top, x1, bottom = t.bbox
            tbl_cov += (x1 - x0) * (bottom - top) / area
            tbl_n += 1
    tbl_cov = min(tbl_cov, 1.0)

    tech_n = len(_TECH.findall(text))
    p = {"chars": chars, "images": sig_imgs, "image_cov": img_cov, "tables": tbl_n,
         "table_cov": tbl_cov, "tech_n": tech_n, "text": text}
    p["label"] = _label(p, biggest)
    return p


def _label(p: dict, biggest_img: float) -> str:
    """Fixed-order ladder: first match wins."""
    if biggest_img >= C["scan_image_coverage"] or (
        p["image_cov"] >= C["scan_low_text_coverage"] and p["chars"] < C["min_chars_text_page"]
    ):
        return "SCAN"
    if p["chars"] < C["min_chars_text_page"] and p["images"] == 0 and p["tables"] == 0:
        return "BLANK"
    if p["image_cov"] >= C["figure_image_coverage"]:
        return "FIGURE"
    if p["table_cov"] >= C["table_area_ratio"]:
        return "TABLE"
    if p["chars"] >= C["technical_min_chars"] and p["tech_n"] / max(p["chars"], 1) >= C["technical_symbol_ratio"]:
        return "TECHNICAL"
    return "PROSE"


def _resolve_blanks(labels: list[str]) -> list[str]:
    """Blank pages inherit the previous real label (leading blanks take the next one)."""
    out, last = labels[:], None
    for i, l in enumerate(out):
        if l == "BLANK":
            out[i] = last or "BLANK"
        else:
            last = l
    nxt = next((l for l in out if l != "BLANK"), "PROSE")
    return [nxt if l == "BLANK" else l for l in out]


def _runs(labels: list[str]) -> list[list]:
    runs = []
    for i, l in enumerate(labels):
        if runs and runs[-1][0] == l:
            runs[-1][2] = i
        else:
            runs.append([l, i, i])
    return runs


def _smooth(labels: list[str]) -> list[list]:
    """Absorb short text-only runs into a neighbour so we don't fragment on noise.
    Figure/table/scan runs are never absorbed (they are loss-critical)."""
    runs = _runs(labels)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lab, s, e) in enumerate(runs):
            if lab in _TEXT_LABELS and (e - s + 1) < C["min_text_segment_pages"]:
                nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
                text_n = [j for j in nbrs if runs[j][0] in _TEXT_LABELS]
                pool = text_n or nbrs
                # prefer the longest neighbour; tie -> earlier one (deterministic)
                j = max(pool, key=lambda k: (runs[k][2] - runs[k][1], -k))
                runs[j][1], runs[j][2] = min(runs[j][1], s), max(runs[j][2], e)
                del runs[i]
                changed = True
                break
        # merge adjacent runs with same label
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged
    return runs


def _aggregate(pages: list[dict], labels: list[str]) -> dict:
    n = max(len(pages), 1)
    chars = sum(p["chars"] for p in pages)
    imgs = sum(p["images"] for p in pages)
    tbls = sum(p["tables"] for p in pages)
    avg = chars // n
    fig_frac = sum(l == "FIGURE" for l in labels) / n
    return {
        "total_tables": tbls,
        "total_images": imgs,
        "images_per_page": round(imgs / n, 2),
        "total_chars": chars,
        "avg_chars_per_page": avg,
        "is_digitally_native": avg >= C["native_min_avg_chars"],
        "has_high_visual_density": fig_frac >= C["doc_figure_fraction"],
        "is_suspected_scan": sum(l == "SCAN" for l in labels) / n >= C["doc_scan_fraction"],
        "scan_fraction": round(sum(l == "SCAN" for l in labels) / n, 4),
        "figure_fraction": round(fig_frac, 4),
        "table_fraction": round(sum(l == "TABLE" for l in labels) / n, 4),
        "table_coverage": round(sum(p["table_cov"] for p in pages) / n, 4),
        "image_coverage": round(sum(p["image_cov"] for p in pages) / n, 4),
        "technical_ratio": round(sum(p["tech_n"] for p in pages) / max(chars, 1), 4),
    }


def _sample(pages: list[dict], start: int, limit: int = 700) -> str:
    idx = sorted({0, len(pages) // 2})
    return "\n".join(f"--- Page {start + i + 1} Sample ---\n{pages[i]['text'][:limit]}" for i in idx)


def profile_full_document(file_path: str):
    """Returns document-level heuristics AND a list of homogeneous segments."""
    pages, errors = [], []
    with pdfplumber.open(file_path) as pdf:
        if len(pdf.pages) == 0:
            return None
        for i, page in enumerate(pdf.pages, start=1):
            try:
                pages.append(_profile_page(page))
            except Exception as e:                       # never silently skip a page
                errors.append({"page": i, "error": type(e).__name__})
                pages.append({"chars": 0, "images": 0, "image_cov": 0.0, "tables": 0,
                              "table_cov": 0.0, "tech_n": 0, "text": "", "label": "BLANK"})

    raw = [p["label"] for p in pages]
    labels = _resolve_blanks(raw)
    segments = []
    for lab, s, e in _smooth(labels):
        seg_pages, seg_labels = pages[s:e + 1], labels[s:e + 1]
        segments.append({
            "start_page": s + 1, "end_page": e + 1, "n_pages": e - s + 1,
            "dominant_label": lab,
            "heuristics": _aggregate(seg_pages, seg_labels),
            "text_sample": _sample(seg_pages, s),
        })

    return {
        "file_name": file_path,
        "total_pages": len(pages),
        "heuristics": _aggregate(pages, labels),
        "text_sample": _sample(pages, 0, 1200),
        "page_labels": labels,
        "page_errors": errors,
        "segments": segments,
    }