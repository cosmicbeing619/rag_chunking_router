# RAG Chunking Strategy Router

Reads a PDF and decides **which chunking strategy to use for which pages**, before the document is split for a RAG pipeline. It does not chunk anything itself. It produces a routing plan with the reasoning behind each decision.

## Why

One chunking method doesn't fit every PDF. Real documents mix prose, tables, figures, and scans:

- Plain text splitting cuts table rows in half.
- It separates figures from their captions.
- It can't read scanned pages at all.

This router profiles each page, groups pages into segments, and assigns the cheapest strategy that preserves the information in each segment. The result is either a **single** strategy for the whole document or a **hybrid** plan with different page ranges routed to different strategies.

## Strategies

| Strategy | Used for |
|---|---|
| `OCR_Pipeline` | Scans and photos with no usable text layer |
| `Multimodal_Layout_Aware` | Figures and charts; keeps images bound to captions and nearby text |
| `Document_Based` | Tables and forms; preserves row/column structure |
| `Semantic` | Dense technical or academic text; splits on topic changes |
| `Recursive` | Ordinary prose; splits by paragraph and sentence |

OCR, Multimodal, and Document_Based are *loss-critical*: a weaker strategy would permanently lose information, so their segments are never merged into text strategies.

## How it works

```
PDF
 -> profile each page (characters, images, tables, math/citation symbols)
 -> filter decorative images (logos, headers, repeated elements)
 -> label each page: SCAN / BLANK / FIGURE / TABLE / TECHNICAL / PROSE
 -> group pages into segments (blank pages inherit, tiny text runs are absorbed)
 -> choose a strategy per segment
 -> absorb tiny soft segments, merge adjacent segments
 -> single or hybrid plan (JSON)
```

### Page labels (first match wins)

| Label | Condition |
|---|---|
| SCAN | One image covers at least 85% of the page, or images cover at least 50% with under 50 characters of text |
| BLANK | Under 50 characters, no relevant images, no tables |
| FIGURE | Relevant images cover at least 1% of the page |
| TABLE | Tables cover at least 20% of the page |
| TECHNICAL | At least 1,200 characters and a math/citation symbol ratio of at least 0.01 |
| PROSE | Everything else |

### Decorative image filter

Images are judged across the whole document. An image is ignored if it is under 30 pt on its short side, has an aspect ratio above 6, repeats on at least 30% of pages (and at least 3), or repeats in the top/bottom 10% margin band. Documents of 4 pages or fewer skip the repetition test.

### Choosing a strategy: four layers

1. **Deterministic pruning** removes strategies that can't apply (no tables, no relevant images, digitally native text). If one option remains, it is chosen and the AI is not called.
2. **Laya AI model** picks between the remaining options and returns a confidence score. Answers are cached so identical input gives identical output.
3. **Deterministic guardrail** takes over if Laya errors or its confidence is below 0.55, using a fixed priority list.
4. **Human review flag** is raised when Laya's top two probabilities are within 0.10.

Each page label restricts which strategies are allowed, so the model can never demote a scan, figure, or table segment to plain text.

All thresholds live in `strategies.py` under `CONFIG`.

## Project structure

```
app.py               Streamlit UI (upload a PDF, view plan, stats, reasoning)
main.py              Decision logic, plan building
pdf_parser.py        Page profiling, image filtering, labeling, segmentation
laya_classifier.py   Option pruning, Laya call, input-keyed cache
strategies.py        Strategy registry and every threshold
requirements.txt
input_docs/          PDFs to process
output/              <name>_strategy.json results and .laya_cache.json
```

## Setup

```bash
git clone https://github.com/cosmicbeing619/rag_chunking_router.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Laya setup: 

# Installation

Requires **Python 3.10+**. 

```bash
# Using pip
pip install laya

# Using uv
uv pip install laya
```

## **Web app**

```bash
streamlit run app.py
```

Upload a PDF to see the decision summary, page map, document stats, and per-segment reasoning. The "All documents" view lists saved results.

**Command line (batch)**

```bash
# put PDFs in input_docs/
python main.py
```

Each PDF writes `output/<name>_strategy.json`.

## Output

```json
{
  "file": "report.pdf",
  "page_labels": ["PROSE", "PROSE", "TABLE", "FIGURE"],
  "plan": {
    "mode": "hybrid",
    "primary_strategy": "Recursive",
    "routes": {"Recursive": ["1-2"], "Document_Based": ["3"], "Multimodal_Layout_Aware": ["4"]},
    "segments": [ ... ],
    "needs_review": false
  }
}
```

Every segment records its layout label, chosen strategy, who decided it (pruning, Laya, cached Laya, or guardrail), confidence, and probabilities.

## Notes on reproducibility

- Page labeling and segmentation are deterministic functions of the PDF and `CONFIG`.
- Laya answers are cached in `output/.laya_cache.json`, keyed by the exact input. After changing the Laya model or the strategy descriptions in `strategies.py`, delete this file and restart the app.

## Limitations

- **Plan only.** The router outputs page ranges and strategies. The chunkers that consume the plan, and a retrieval evaluation, are not built yet.
- **No measured accuracy yet.** Thresholds were set by hand. A labeled test set and a comparison of retrieval quality against single-strategy chunking are the next step.
- **Vector-drawn charts are not detected.** Charts drawn as lines and curves (for example matplotlib or Excel exports) are not embedded images, so those pages are labeled as prose.
- **Borderless tables are not detected.** Table detection is limited to ruled tables to avoid false positives.
- **Full-page figures can be labeled SCAN.** The 85% single-image rule ignores text, so slides or full-page diagrams in digital PDFs may be routed to OCR.
- **Searchable scans.** Scans with an embedded OCR text layer can look like digital documents, and the quality of that layer is not checked.
- **Mixed pages get one label.** A page that is half table and half text is labeled by rule priority, not split by region.
- **Repetition-based logo detection is weak on very short PDFs** (4 pages or fewer).
- **PDF only.** Other formats must be converted first.

