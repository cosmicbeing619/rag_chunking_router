"""
Stackable Chunking Registry + single source of truth for every threshold.
To add a strategy: add it to CHUNKING_REGISTRY, then to STRATEGY_PRIORITY,
and (optionally) to LOSS_CRITICAL / SOFT_STRATEGIES / ALLOWED_BY_LABEL.
"""

CHUNKING_REGISTRY = {
    "OCR_Pipeline": (
        "Document is a scan or physical photograph. Very low extractable text or "
        "consists of full-page scanned sheets where OCR is required to extract handwriting or text."
    ),
    "Multimodal_Layout_Aware": (
        "Digitally native PDF with multiple embedded figures, architecture diagrams, loss curves, or charts. "
        "Requires extracting and binding image artifacts/captions to neighboring text chunks so visual context is preserved."
    ),
    "Document_Based": (
        "Structured tables, forms, or balance sheets dominate the document. "
        "Requires row/column structural chunking to prevent table distortion."
    ),
    "Semantic": (
        "Dense, continuous technical text, academic papers, or formulas with minimal/no diagrams. "
        "Requires embedding-distance chunking on conceptual boundaries."
    ),
    "Recursive": (
        "Standard sequential prose, documentation, manuals, or fiction with consistent paragraph structure."
    ),
}

# Lower index = higher priority (used for deterministic tie-breaks).
STRATEGY_PRIORITY = ["OCR_Pipeline", "Multimodal_Layout_Aware", "Document_Based", "Semantic", "Recursive"]

# Choosing a weaker strategy for these loses information -> never merged away in hybrid plans.
LOSS_CRITICAL = {"OCR_Pipeline", "Multimodal_Layout_Aware", "Document_Based"}

# Cheap-to-confuse text strategies; tiny segments of these get absorbed to avoid fragmentation.
SOFT_STRATEGIES = {"Semantic", "Recursive"}

# Page-label -> strategies Laya is allowed to pick between for a segment of that label.
# Laya only arbitrates genuinely ambiguous choices; it can never demote a figure/table/scan segment to plain text.
ALLOWED_BY_LABEL = {
    "SCAN": ["OCR_Pipeline"],
    "FIGURE": ["Multimodal_Layout_Aware", "Document_Based"],
    "TABLE": ["Document_Based", "Multimodal_Layout_Aware"],
    "TECHNICAL": ["Semantic", "Recursive"],
    "PROSE": ["Recursive", "Semantic"],
}

CONFIG = {
    # page level
    "min_chars_text_page": 50,
    "min_image_area_ratio": 0.01,      # ignore logos/icons
    "scan_image_coverage": 0.85,       # one image covers >=85% of page
    "scan_low_text_coverage": 0.50,
    "figure_image_coverage": 0.01,
    "table_area_ratio": 0.20,
    "table_min_rows": 2,
    "table_min_cols": 2,
    "technical_symbol_ratio": 0.01,
    "technical_min_chars": 1200,
    "native_min_avg_chars": 200,
    # segmentation
    "min_text_segment_pages": 2,       # PROSE/TECHNICAL runs shorter than this get absorbed
    "min_soft_segment_pages": 3,       # Semantic/Recursive segments shorter than this get absorbed
    # doc level
    "doc_scan_fraction": 0.50,
    "doc_figure_fraction": 0.25,
    # routing
    "laya_min_confidence": 0.55,
    "review_margin": 0.10,             # flag for human review if Laya top-2 gap is below this
        # decorative-image filter (logos, headers, watermarks, rules)
    "deco_repeat_fraction": 0.30,      # same image on >=30% of pages ...
    "deco_repeat_min_pages": 3,        # ... and at least this many pages
    "deco_repeat_max_area": 0.15,      # repeat rule only applies to images smaller than this (page fraction)
    "deco_margin_band": 0.10,          # top/bottom 10% of the page counts as header/footer band
    "deco_margin_min_repeats": 2,      # margin image repeated on >=2 pages is decorative
    "deco_min_side_pt": 30,            # shorter side under this = icon/rule
    "deco_max_aspect": 6.0,            # long/short side above this = divider/banner line
    "short_doc_pages": 4,              # docs this short: no repetition test, margin + size only
}