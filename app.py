"""
Streamlit front-end for the chunking router.   Run:  streamlit run app.py

This is a thin view over main.py. The CLI (`python main.py`) does not need Streamlit.
Uploaded PDFs are saved to input_docs/ and results to output/, exactly like the CLI.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

os.chdir(Path(__file__).resolve().parent)          # main.py uses relative paths

from main import INPUT_DIR, OUTPUT_DIR, setup_folders, analyze_pdf, save_result  # noqa: E402
from laya_classifier import DocumentClassifier                                    # noqa: E402
from strategies import CONFIG as C, LOSS_CRITICAL                                 # noqa: E402

LABELS = {
    "SCAN": ("#d9534f", f"a single image covers ≥{C['scan_image_coverage']:.0%} of the page, or images cover "
                        f"≥{C['scan_low_text_coverage']:.0%} with <{C['min_chars_text_page']} text characters"),
    "FIGURE": ("#8e44ad", f"significant images cover ≥{C['figure_image_coverage']:.0%} of the page"),
    "TABLE": ("#e67e22", f"tables (≥{C['table_min_rows']} rows × ≥{C['table_min_cols']} cols) cover "
                         f"≥{C['table_area_ratio']:.0%} of the page"),
    "TECHNICAL": ("#2980b9", f"≥{C['technical_min_chars']} characters with a math/citation symbol ratio "
                             f"≥{C['technical_symbol_ratio']}"),
    "PROSE": ("#27ae60", "none of the higher rules matched (plain running text)"),
    "BLANK": ("#95a5a6", "almost no text, images or tables"),
}
STRATEGY_COLORS = {
    "OCR_Pipeline": "#d9534f", "Multimodal_Layout_Aware": "#8e44ad", "Document_Based": "#e67e22",
    "Semantic": "#2980b9", "Recursive": "#27ae60",
}
SOURCE_COLOR = {
    "Deterministic Pruning": "blue", "Laya AI Model": "green", "Laya AI Model (cached)": "green",
    "Deterministic Guardrail": "orange", "merged": "gray",
}


# ----------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading Laya model...")
def get_classifier() -> DocumentClassifier:
    return DocumentClassifier(cache_path=str(OUTPUT_DIR / ".laya_cache.json"))


def load_results() -> list[dict]:
    out = []
    for p in sorted(OUTPUT_DIR.glob("*_strategy.json"), key=lambda x: x.name):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def pages_str(s: dict) -> str:
    return str(s["start_page"]) if s["start_page"] == s["end_page"] else f"{s['start_page']}-{s['end_page']}"


def explain(s: dict) -> str:
    """Plain-English reasoning built only from fields present in the JSON."""
    lab = s["layout_label"]
    lines = [f"1. **Layout:** pages {pages_str(s)} were labeled **{lab}** because {LABELS[lab][1]}."]
    src = s["routing_source"]
    opts = s.get("options") or []
    if src == "Deterministic Pruning":
        lines.append(f"2. **Pruning:** only one strategy was valid ({', '.join(opts) or s['strategy']}), "
                     "so Laya was not consulted.")
    elif src.startswith("Laya AI Model"):
        lines.append(f"2. **Pruning:** Laya chose between {', '.join(opts) if opts else 'the remaining options'}.")
        conf = s.get("confidence_score")
        lines.append(f"3. **Laya:** picked **{s['strategy']}** with confidence "
                     f"{conf if conf is not None else 'n/a'} (threshold {C['laya_min_confidence']}), "
                     f"so its answer was accepted{' (from cache)' if 'cached' in src else ''}.")
        if s.get("needs_review"):
            lines.append(f"4. **Review flag:** the top two probabilities were within {C['review_margin']}.")
    elif src == "Deterministic Guardrail":
        raw, rc = s.get("ai_raw_choice"), s.get("ai_raw_confidence")
        lines.append(f"2. **Pruning:** Laya was allowed to choose between {', '.join(opts) if opts else 'the remaining options'}.")
        lines.append(f"3. **Laya:** answered **{raw}** with confidence {rc} — below {C['laya_min_confidence']} "
                     "or an error — so it was **overridden**.")
        lines.append(f"4. **Guardrail:** {s.get('routing_note', '')} → **{s['strategy']}**.")
    elif src == "merged":
        lines.append(f"2. **Merged:** adjacent segments with the same strategy were joined. Confidence and "
                     "probabilities shown belong to the first piece only.")
    if s.get("absorbed_from"):
        lines.append(f"• This small segment was originally **{s['absorbed_from']}** and was absorbed into "
                     f"**{s['strategy']}** to avoid fragmenting the document "
                     f"(<{C['min_soft_segment_pages']} pages).")
    return "\n\n".join(lines)


def strip_html(colors: list[str], titles: list[str]) -> str:
    cells = "".join(f'<div title="{t}" style="flex:1;min-width:3px;height:22px;background:{c}"></div>'
                    for c, t in zip(colors, titles))
    return f'<div style="display:flex;gap:1px;margin-bottom:6px">{cells}</div>'


def legend_html(mapping: dict) -> str:
    return " ".join(f'<span style="display:inline-block;width:10px;height:10px;background:{c};'
                    f'margin:0 4px 0 10px"></span><small>{k}</small>' for k, c in mapping.items())


# ----------------------------------------------------------------- result view
def render_result(final: dict):
    plan, stats = final["plan"], final["document_stats"]
    segs = plan["segments"]
    n_pages = len(final["page_labels"])

    st.subheader(final["file"])
    st.caption(f"{n_pages} pages · sha256 {final['file_sha256'][:16]}…")

    # --- decision
    st.markdown("### Decision")
    overrides = sum(s["routing_source"] == "Deterministic Guardrail" for s in segs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", plan["mode"].upper())
    c2.metric("Primary strategy", plan["primary_strategy"])
    c3.metric("Segments", len(segs))
    c4.metric("Needs review", "Yes" if plan["needs_review"] else "No")
    if plan["mode"] == "hybrid":
        st.info("Hybrid: different page ranges use different chunking strategies.")
    else:
        st.info("Single: one chunking strategy for the whole document.")
    if overrides:
        st.warning(f"{overrides} segment(s) were decided by the deterministic guardrail because Laya's "
                   "confidence was too low or it errored.")
    if final.get("page_errors"):
        st.error(f"pdfplumber failed on pages: {[e['page'] for e in final['page_errors']]}")

    rows = []
    for strat, ranges in plan["routes"].items():
        rows.append({"Strategy": strat, "Pages": ", ".join(ranges),
                     "# pages": sum(s["n_pages"] for s in segs if s["strategy"] == strat),
                     "Loss-critical": "yes" if strat in LOSS_CRITICAL else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # --- timelines
    st.markdown("### Page map")
    lab = final["page_labels"]
    st.caption("Detected layout per page")
    st.markdown(strip_html([LABELS[l][0] for l in lab], [f"p{i}: {l}" for i, l in enumerate(lab, 1)])
                + legend_html({k: v[0] for k, v in LABELS.items() if k != "BLANK"}), unsafe_allow_html=True)
    per_page = {}
    for s in segs:
        for p in range(s["start_page"], s["end_page"] + 1):
            per_page[p] = s["strategy"]
    st.caption("Assigned chunking strategy per page")
    st.markdown(strip_html([STRATEGY_COLORS.get(per_page.get(i), "#999") for i in range(1, n_pages + 1)],
                           [f"p{i}: {per_page.get(i)}" for i in range(1, n_pages + 1)])
                + legend_html(STRATEGY_COLORS), unsafe_allow_html=True)

    # --- stats
    st.markdown("### Document stats")
    a = st.columns(5)
    a[0].metric("Pages", n_pages)
    a[1].metric("Avg chars/page", stats["avg_chars_per_page"])
    a[2].metric("Tables", stats["total_tables"])
    a[3].metric("Significant images", stats["total_images"])
    a[4].metric("Technical ratio", stats["technical_ratio"])
    b = st.columns(5)
    b[0].metric("Scan pages", f"{stats['scan_fraction']:.0%}")
    b[1].metric("Figure pages", f"{stats['figure_fraction']:.0%}")
    b[2].metric("Table pages", f"{stats['table_fraction']:.0%}")
    b[3].metric("Digitally native", "yes" if stats["is_digitally_native"] else "no")
    b[4].metric("High visual density", "yes" if stats["has_high_visual_density"] else "no")

    # --- segments + reasoning
    st.markdown("### Segments and reasoning")
    for i, s in enumerate(segs, 1):
        src = s["routing_source"]
        with st.expander(f"Segment {i} · pages {pages_str(s)} · {s['strategy']}", expanded=len(segs) <= 4):
            m = st.columns(4)
            m[0].metric("Layout label", s["layout_label"])
            m[1].metric("Strategy", s["strategy"])
            conf = s.get("confidence_score")
            m[2].metric("Laya confidence", "—" if conf is None else conf)
            m[3].markdown(f"**Decided by**\n\n:{SOURCE_COLOR.get(src, 'gray')}[{src}]")
            st.markdown(explain(s))
            if s.get("probabilities"):
                st.caption("Laya probabilities")
                st.bar_chart(pd.Series(s["probabilities"]))

    dl = plan.get("document_level_classification") or {}
    if dl and plan["mode"] == "hybrid":
        with st.expander("Audit: whole-document classification (does not affect the plan)"):
            st.json(dl)

    with st.expander("Raw JSON"):
        st.json(final)
    st.download_button("Download JSON", json.dumps(final, indent=2, sort_keys=True),
                       file_name=f"{Path(final['file']).stem}_strategy.json", mime="application/json")


# ----------------------------------------------------------------- pages
def page_analyze():
    st.title("Chunking strategy router")
    up = st.file_uploader("Upload a PDF", type="pdf")
    if up is None:
        st.caption("Upload a PDF to get a single or hybrid chunking plan with stats and reasoning.")
        return
    data = up.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("digest") != digest:
        setup_folders()
        path = INPUT_DIR / Path(up.name).name
        path.write_bytes(data)
        clf = get_classifier()
        try:
            with st.spinner("Profiling pages and classifying segments..."):
                final = analyze_pdf(path, clf)
        except Exception as e:
            st.error(f"Could not process this PDF ({type(e).__name__}). It may be encrypted or corrupt.")
            return
        if not final:
            st.error("The PDF has no pages.")
            return
        save_result(final)
        clf.save_cache()
        st.session_state["digest"], st.session_state["final"] = digest, final
    render_result(st.session_state["final"])


def page_all_docs():
    st.title("All processed documents")
    results = load_results()
    if not results:
        st.info(f"No results in '{OUTPUT_DIR}/' yet. Upload a PDF or run `python main.py`.")
        return
    rows = []
    for r in results:
        p = r["plan"]
        rows.append({
            "File": r["file"], "Pages": len(r["page_labels"]), "Mode": p["mode"],
            "Primary strategy": p["primary_strategy"], "Strategies": ", ".join(p["routes"]),
            "Segments": len(p["segments"]),
            "Guardrail overrides": sum(s["routing_source"] == "Deterministic Guardrail" for s in p["segments"]),
            "Needs review": p["needs_review"],
        })
    df = pd.DataFrame(rows)
    f1, f2 = st.columns(2)
    modes = f1.multiselect("Mode", sorted(df["Mode"].unique()), default=sorted(df["Mode"].unique()))
    strats = f2.multiselect("Primary strategy", sorted(df["Primary strategy"].unique()),
                            default=sorted(df["Primary strategy"].unique()))
    view = df[df["Mode"].isin(modes) & df["Primary strategy"].isin(strats)]
    st.dataframe(view, hide_index=True)
    st.caption(f"{len(view)} of {len(df)} documents")
    if view.empty:
        return
    choice = st.selectbox("Open a document", view["File"].tolist())
    st.divider()
    render_result(next(r for r in results if r["file"] == choice))


st.set_page_config(page_title="Chunking router", layout="wide")
page = st.sidebar.radio("View", ["Analyze a PDF", "All documents"])
(page_analyze if page == "Analyze a PDF" else page_all_docs)()