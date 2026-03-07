# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run all tests (no GPU needed — MockModel)
python -m pytest test.py -v

# Run a single test class
python -m pytest test.py -v -k "TestV1"

# Invoice demo — run all 3 versions and compare
python invoice_demo.py

# Run one specific version
python invoice_demo.py --version v1
python invoice_demo.py --version v2
python invoice_demo.py --version v3 --no-mlm   # skip MLM scorer

# Medical record demo
python med_demo.py --version v1 --trials 5
```

## Architecture Overview

The project implements LLM extraction confidence via **causal perturbation testing** — a generative model is turned into an empirical probabilistic extractor without retraining.

There are **three progressively sophisticated versions**, each a complete standalone pipeline. All share the same perturbation-testing stages (2–4); they differ only in how the **anchor is found** and how the **confidence score is combined**.

### Directory Structure

```
shared/           # Used by all 3 versions
  schemas.py      # All dataclasses (ExtractionResult, AnchorResult, etc.)
  models.py       # HFExtractionModel — HuggingFace wrapper with gradient support
  value_generator.py  # Type-aware random value generation (currency, date, etc.)
  localizer.py    # Stage 2: value injection + re-extraction trials
  anchor_validator.py # Stage 3: context corruption + re-extraction trials

v1_basic/         # Pure text — no model internals, works with any LLM
  anchor_finder.py    # HeuristicAnchorFinder: value-match → sentence masking
  confidence.py       # ConfidenceScorer: sqrt(loc * anchor)  [2-component]
  extractor.py        # BasicConfidenceExtractor

v2_intermediate/  # Gradient attribution — open-weight HF models only
  anchor_finder.py    # GradientAnchorFinder: ||grad·emb||₂ per token
  confidence.py       # WeightedConfidenceScorer: w1*attn + w2*loc + w3*anchor  [3-component]
  extractor.py        # IntermediateConfidenceExtractor

v3_advanced/      # Probabilistic — adds logit entropy + MLM perplexity
  models.py           # AdvancedHFExtractionModel: adds extract_with_logits() + MC dropout
  anchor_finder.py    # AdvancedAnchorFinder: gradient + MLM re-scoring
  entropy_scorer.py   # compute_generation_entropy(): 1 − normalised_entropy
  mlm_scorer.py       # MLMAnchorScorer: BERT pseudo-perplexity of anchor text
  confidence.py       # AdvancedConfidenceScorer: 4-component + bootstrap CI
  extractor.py        # AdvancedConfidenceExtractor
```

### Pipeline Stages (all versions)

| Stage | File | v1 | v2 | v3 |
|-------|------|----|----|----|
| 1 – Extract value | `shared/models.py` | `.extract()` | `.extract()` | `.extract_with_logits()` |
| 2 – Find anchor | version-specific | value-match / sentence-mask | gradient attribution | gradient + MLM re-score |
| 3 – Localization test | `shared/localizer.py` | same | same | same |
| 4 – Anchor validation | `shared/anchor_validator.py` | same | same | same |
| 5 – Confidence | version-specific | `sqrt(L*A)` | `w1*C + w2*L + w3*A` | `w1*C + w2*L + w3*A + w4*E` + bootstrap CI |

### Confidence Formula by Version

- **v1**: `sqrt(localization_score × anchor_validation_score)` — geometric mean, Laplace-smoothed
- **v2**: `0.20*attention_concentration + 0.40*localization + 0.40*anchor_validation`
- **v3**: `0.15*attention + 0.35*localization + 0.35*anchor_validation + 0.15*entropy_score`
  - anchor component is MLM-modulated: `0.7*anchor_validation + 0.3*mlm_score`
  - plus bootstrap 95% CI from trial-level resampling

### Key Data Flow

```
extractor.extract(text, field)
  → model.extract() / extract_with_logits()     → value: str
  → anchor_finder.find()                         → AnchorResult + concentration [+ mlm_score]
  → localizer.test()                             → LocalizationResult
  → anchor_validator.validate()                  → AnchorValidationResult
  → [entropy_scorer.compute_generation_entropy()] → entropy_score (v3 only)
  → confidence_scorer.compute()                  → ConfidenceBreakdown
  → ExtractionResult (all stages combined)
```

### ConfidenceBreakdown fields

`ConfidenceBreakdown` in `shared/schemas.py` has optional fields so v1/v2 results stay compatible:
- `attention_score` — set by v2/v3 (gradient attribution concentration)
- `entropy_score` — set by v3 (logit entropy)
- `mlm_perplexity` — set by v3 (anchor perplexity from distilbert)
- `confidence_interval` — set by v3 (bootstrap 95% CI as `(lo, hi)`)

### Confidence Thresholds

| Score | Interpretation |
|-------|---------------|
| ≥ 0.80 | HIGH — reliable |
| ≥ 0.55 | MEDIUM — review recommended |
| ≥ 0.30 | LOW — manual verification advised |
| < 0.30 | VERY LOW — do not trust |

## Model Loading Notes

`HFExtractionModel` (in `shared/models.py`) auto-detects seq2seq vs causal LM. On Windows with CUDA, it loads to CPU first then moves to GPU to avoid OS error 1455. Default model: `Qwen/Qwen2.5-1.5B-Instruct`. 4-bit NF4 quantization via bitsandbytes is optional and guarded by `_bnb_cuda_ok()`.

`AdvancedHFExtractionModel` (in `v3_advanced/models.py`) extends this with `extract_with_logits()` (captures per-token logits via `output_scores=True`) and `extract_mc()` (Monte Carlo dropout).

The MLM scorer in `v3_advanced/mlm_scorer.py` lazily loads `distilbert-base-uncased` on first use.
