# LLM Confidence Extractor

A Python library that adds **confidence measurement** to LLM-based information extraction — similar to how traditional OCR or discriminative models report localisation confidence, but for generative LLMs.

---

## Motivation

Generative LLMs are non-deterministic and produce no intrinsic confidence score. Traditional discriminative models (classic OCR, NER taggers) compensate by outputting a **localisation span** + **confidence percentage** for each extracted entity. This library replicates that behaviour for any instruction-following LLM.

---

## How It Works

The pipeline runs four steps:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1 – Extract                                                    │
│    Run the LLM.  Capture the extracted value + attention weights.    │
├──────────────────────────────────────────────────────────────────────┤
│  Step 2 – Find Anchor                                                │
│    Aggregate per-token attention across all generation steps and     │
│    across the last N transformer layers.  The contiguous high-       │
│    attention region in the source text is the "anchor" — the        │
│    paragraph / sentence the model read to derive the answer.         │
│    Fallback (API models): use the sentence containing the value.     │
├──────────────────────────────────────────────────────────────────────┤
│  Step 3 – Localization Test                                          │
│    Replace the value in the source text with K different random      │
│    values of the same type (numeric → random number, date → random   │
│    date, …).  Re-run extraction each time.  If the model extracts    │
│    those random values, the location is confirmed.                   │
│    localization_accuracy = hits / K                                  │
├──────────────────────────────────────────────────────────────────────┤
│  Step 4 – Anchor Validation                                          │
│    Replace the anchor *context* (everything except the value) with   │
│    random words.  Re-run extraction.  If the extracted value changes │
│    or the model fails, the anchor context was genuinely needed.      │
│    anchor_sensitivity = trials_where_extraction_changed / K          │
├──────────────────────────────────────────────────────────────────────┤
│  Step 5 – Confidence Score                                           │
│    overall = w₁·attention_concentration                              │
│            + w₂·localization_accuracy                                │
│            + w₃·anchor_sensitivity                                   │
│    Default weights: 0.20 / 0.40 / 0.40                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone <repo>
cd llm_confidence_extractor
pip install -r requirements.txt
```

---

## Quick Start

```python
from llm_confidence_extractor.backends import HuggingFaceLLMBackend
from llm_confidence_extractor.pipeline import ConfidenceExtractorPipeline

# Load any HuggingFace instruction-tuned model
backend = HuggingFaceLLMBackend("Qwen/Qwen2.5-0.5B-Instruct")

pipeline = ConfidenceExtractorPipeline(
    backend=backend,
    n_localization_trials=5,   # more trials = more reliable confidence
    n_anchor_trials=5,
)

text = """
Invoice #INV-2024-0892  |  Date: March 15, 2024
Client: Acme Corporation
Total Due: $22,242.50   (Tax 8.5% included)
Payment due within 30 days.
"""

result = pipeline.extract(text, field_name="total amount due")
print(result.summary())
# Field:       total amount due
# Value:       $22,242.50
# Confidence:  87.0%
#   ├─ Attention concentration: 73.0%
#   ├─ Localization accuracy:   100.0%
#   └─ Anchor sensitivity:      80.0%
# Anchor:      "Total Due: $22,242.50   (Tax 8.5% included)"
```

### Extract multiple fields at once

```python
results = pipeline.extract_many(
    text=text,
    fields={
        "date":    "invoice date",
        "client":  "client or company name",
        "total":   "total amount due",
        "tax":     "tax rate percentage",
    },
)

for key, r in results.items():
    print(f"{key:10s}  {r.extracted_value:20s}  conf={r.confidence.overall:.0%}")
```

---

## Project Structure

```
llm_confidence_extractor/
├── requirements.txt
├── examples/
│   └── demo.py                    ← runnable demo (invoice + contract)
├── tests/
│   └── test_pipeline.py           ← unit tests (no GPU needed)
└── src/
    └── llm_confidence_extractor/
        ├── __init__.py
        ├── models.py              ← dataclasses (ExtractionResult, etc.)
        ├── pipeline.py            ← ConfidenceExtractorPipeline (main entry)
        ├── anchor_finder.py       ← Step 2: attention → anchor span
        ├── localizer.py           ← Step 3: value perturbation test
        ├── anchor_validator.py    ← Step 4: context perturbation test
        ├── confidence_scorer.py   ← Step 5: weighted confidence
        ├── perturbation.py        ← random value/context generators
        └── backends/
            ├── base.py            ← BaseLLMBackend (abstract)
            └── huggingface.py     ← HuggingFaceLLMBackend
```

---

## Running the Demo

```bash
# Small model (fast, works on CPU)
python examples/demo.py --model Qwen/Qwen2.5-0.5B-Instruct --trials 5 --cpu

# Larger model (better accuracy, needs GPU)
python examples/demo.py --model meta-llama/Llama-3.2-1B-Instruct --trials 10

# Suppress per-step verbosity
python examples/demo.py --quiet
```

---

## Running Tests

```bash
cd llm_confidence_extractor
python -m pytest tests/ -v
```

Tests use a `MockBackend` — no model download required.

---

## Supported Models

Any HuggingFace causal LM with a chat template works out of the box:

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Recommended for development |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Good accuracy / speed balance |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Requires HuggingFace access token |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Strong instruction following |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | High accuracy, needs ~14 GB VRAM |

---

## Adding a Custom Backend

Implement `BaseLLMBackend` to wrap any LLM (OpenAI, Anthropic, local GGUF, etc.):

```python
from llm_confidence_extractor.backends.base import BaseLLMBackend
import numpy as np

class MyAPIBackend(BaseLLMBackend):
    def supports_attention(self) -> bool:
        return False  # API models don't expose attention

    def extract(self, text, field, return_attention=True, **kwargs):
        value = my_api_call(text, field)   # your implementation
        return value, None, []             # no attention weights

    def _build_prompt(self, text, field, system_prompt=None):
        return f"Text: {text}\nExtract: {field}"
```

When `supports_attention()` returns `False`, the pipeline automatically falls back
to heuristic anchor finding and still runs both perturbation tests.

---

## Configuration Reference

```python
ConfidenceExtractorPipeline(
    backend                = backend,
    n_localization_trials  = 5,      # Step 3 trial count
    n_anchor_trials        = 5,      # Step 4 trial count
    anchor_top_k_tokens    = 20,     # Attention tokens used for anchor span
    anchor_context_window  = 100,    # Chars added each side of top-K span
    attention_weight       = 0.20,   # Confidence weight (must sum to 1)
    localization_weight    = 0.40,
    anchor_weight          = 0.40,
    seed                   = 42,     # RNG seed for reproducibility
)
```

---

## Interpreting Confidence Scores

| Score | Interpretation |
|---|---|
| ≥ 0.85 | High confidence — extraction is well-grounded |
| 0.65–0.85 | Medium confidence — review recommended |
| 0.50–0.65 | Low confidence — manual verification advised |
| < 0.50 | Very low — likely incorrect or hallucinated |

The three sub-scores help diagnose *why* confidence is low:

- **Low attention concentration** → anchor is diffuse; model spread attention widely
- **Low localization accuracy** → model is not reading the value from the expected position
- **Low anchor sensitivity** → model extracts the same value regardless of context (possible position bias or hallucination)