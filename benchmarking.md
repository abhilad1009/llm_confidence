# Benchmarking the LLM Confidence Extractor: A Research Plan

## 1. Research Framing and Contribution

### Core Thesis

Existing LLM uncertainty methods rely on model internals (logprobs, embeddings) or stochastic sampling. This system introduces **perturbation-based confidence estimation**: instead of asking "how certain is the model?" it asks "does the model's behaviour change when we alter the source in controlled ways?" This is a fundamentally different and complementary paradigm — it measures *functional grounding* rather than *probabilistic belief*.

### Novelty Claims (What Reviewers Will Look For)

1. **Perturbation-based localization testing** — replacing the extracted value with type-matched random values to verify the model actually reads from the right location, not hallucinating or pattern-matching.
2. **Anchor sensitivity analysis** — destroying the surrounding context to test whether the model relies on genuine semantic understanding of the passage.
3. **Black-box compatibility** — unlike logprob or attention methods, Steps 3 and 4 work with *any* LLM with zero access to internals.
4. **Composite scoring** that separates *where the model looked* from *whether the answer is right* — a distinction no existing method makes.

### Suggested Paper Title

> "Perturbation-Based Confidence Estimation for LLM Information Extraction: Measuring Functional Grounding Without Model Internals"

---

## 2. What to Benchmark Against (Baselines)

All baselines use HuggingFace open-weight models directly. No paid API calls.

### 2.1 Token-Level Baselines (Require Logprob Access)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Mean Token Log-Probability** | Average log-prob of all output tokens | Simplest and most common LLM confidence proxy |
| **Min Token Log-Probability** | Lowest log-prob token in the answer | Captures "weakest link" uncertainty |
| **Normalized Sequence Probability** | Length-normalized product of all token probs | Controls for answer length bias |

**Implementation**: Call HF model with `generate(..., output_scores=True, return_dict_in_generate=True)`, extract per-token log-probs from `outputs.scores`.

### 2.2 Sampling-Based Baseline (Black-Box Compatible)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Self-Consistency (SC)** | Sample N=10 outputs (temperature=0.7), score = fraction agreeing with majority answer | Gold-standard black-box method (Wang et al., 2023) |

**Implementation**: Call `model.extract()` N=10 times at temperature 0.7, normalize answers, compute majority agreement ratio.

### 2.3 Attention-Based Baseline (Partial Overlap With This System)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Attention Entropy** | Concentration of gradient attribution over source tokens (1 − normalised entropy) | Isolates the contribution of Step 2 (anchor finding); shows perturbation steps add value beyond attention alone |

**Implementation**: Reuse `GradientAnchorFinder` from `v2_intermediate/anchor_finder.py`. The `concentration` score returned is already `1 − normalised_entropy` — use it directly as confidence.

### 2.4 Anchor Visualization

For v2 and v3, every extraction result includes per-token attribution scores in `AnchorResult.token_attributions`. The benchmark **must display these as a token-level heatmap** in the output report — confirming the gradient-based anchor finder is correctly identifying which source tokens drove the extraction.

- **Text heatmap**: ANSI-colored token display, intensity proportional to attribution score
- **Bar chart**: matplotlib bar plot of score vs token index, annotated with token text
- Both take `list[TokenAttribution]` from `AnchorResult.token_attributions`
- If `token_attributions` is empty or all-zeros for v2/v3, this indicates a bug in `forward_with_grad()` → offset_mapping alignment

---

## 3. Datasets

### 3.1 Evaluation Datasets

| Dataset | Domain | Task | Why It Matters |
|---------|--------|------|----------------|
| **SQuAD 2.0** | Wikipedia | Extractive QA + unanswerable | Has unanswerable questions → tests if confidence is low when model should abstain |
| **FUNSD** | Scanned forms | Form key-value extraction | Noisy OCR + layout; tests robustness |
| **CORD** | Receipts | Receipt field extraction | Structured extraction with positional reasoning |

All loaded via the HuggingFace `datasets` library:
- `rajpurkar/squad_v2`
- `nielsr/funsd`
- `naver-clova-ix/cord-v2`

### 3.2 Dataset Scale

| Dataset | Test Set | Dev Set (weight tuning) |
|---------|----------|------------------------|
| SQuAD 2.0 | 500 examples (250 answerable, 250 unanswerable) | 100 examples |
| FUNSD | 200 examples | 80 examples |
| CORD | 200 examples | 80 examples |

---

## 4. Evaluation Metrics

A confidence score is only useful if it *predicts whether the extraction is correct*. The ground-truth label for each example is binary: **correct extraction** (1) vs **incorrect extraction** (0), determined by exact match or fuzzy match against gold labels.

### 4.1 Discrimination Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **AUROC** | Overall ranking quality — can the score separate correct from incorrect? | Higher is better; > 0.80 is strong |
| **Spearman ρ** | Rank correlation between confidence and a continuous correctness measure (e.g., F1 per example) | Higher is better |

### 4.2 Calibration Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Expected Calibration Error (ECE)** | Mean gap between confidence and actual accuracy in equal-width bins | Lower is better; < 0.05 is well-calibrated |
| **Brier Score** | Mean squared error between confidence and binary correctness | Lower is better; combines calibration + discrimination |
| **Reliability Diagram** | Visual: plot confidence bins (x) vs actual accuracy (y) | Should follow the diagonal |

### 4.3 Selective Prediction Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Risk-Coverage Curve** | At each coverage level (fraction of examples kept), what is the error rate? | Lower error at each coverage level |
| **AUC of Risk-Coverage** | Area under the above curve | Lower is better |
| **Accuracy@K% Coverage** | Accuracy when keeping only the top K% most confident predictions | Higher is better; report K = 90%, 80%, 70%, 50% |

---

## 5. Experimental Design

### 5.1 Models

All experiments use HuggingFace open-weight models directly:

- **Qwen/Qwen3-0.6B** — default model (already used in the codebase). Runs on CPU or GPU.
- **Qwen/Qwen3-1.7B** — optional larger variant for comparison.

No closed-source or paid API models.

### 5.2 Ablation Studies

| Ablation | Config | What It Shows |
|----------|--------|---------------|
| **v1_baseline** | `BasicConfidenceExtractor` (no attention) | Heuristic-only baseline |
| **perturbation_only** | v2 with `attention_weight=0` (w2=0.5, w3=0.5) | Value of perturbation steps without attention |
| **no_localization** | v2 with `localization_weight=0` (renormalize) | Value of value-replacement testing |
| **no_anchor_validation** | v2 with `anchor_validation_weight=0` (renormalize) | Value of context destruction testing |
| **full_v2** | Default v2 weights (0.20, 0.40, 0.40) | Full 3-component system |
| **full_v3** | Default v3 weights (0.15, 0.35, 0.35, 0.15) | Full 4-component system with entropy + MLM |

### 5.3 Sensitivity Analysis

- **K (number of perturbation trials)**: Sweep K = {3, 5, 10, 15, 20}. Plot AUROC vs K. Find the elbow point.
- **Random seed robustness**: Run the full pipeline 3 times with different random perturbation seeds; report variance.

### 5.4 Qualitative Analysis

Include 5–10 worked examples showing:
- Cases where perturbation confidence is low but logprob confidence is high (hallucinated but fluent)
- Cases where perturbation confidence is high but the answer is wrong (failure mode analysis)
- Token attribution heatmaps for v2/v3 showing which source tokens drove each extraction
- Reliability diagrams overlaid for all methods

---

## 6. Implementation Plan

### Phase 0: Anchor Visualization (Priority)

```
Task 0.1  Create benchmark/visualize.py
  - render_attribution_heatmap(source_text, token_attributions): ANSI text heatmap
  - plot_attribution_bar(token_attributions): matplotlib bar chart
  - Verify v2/v3 anchor finder produces non-empty token_attributions

Task 0.2  Surface in demo scripts
  - invoice_demo.py: show heatmap for v2/v3 after each extraction
  - med_demo.py: same
```

### Phase 1: Dataset Preparation

```
Task 1.1  benchmark/data/squad.py
  - load_dataset("rajpurkar/squad_v2"), map to BenchmarkExample
  - Stratify: 250 answerable + 250 unanswerable from validation set
  - Dev/test split

Task 1.2  benchmark/data/funsd.py
  - load_dataset("nielsr/funsd"), reconstruct doc text from word tokens
  - Map linked key-value pairs to BenchmarkExample

Task 1.3  benchmark/data/cord.py
  - load_dataset("naver-clova-ix/cord-v2"), parse ground_truth.gt_parse JSON
  - Map receipt fields to BenchmarkExample
```

### Phase 2: Baseline Methods

```
Task 2.1  benchmark/baselines/logprob.py
  - LogprobConfidence: mean, min, length-normalized logprob via output_scores=True

Task 2.2  benchmark/baselines/self_consistency.py
  - SelfConsistencyConfidence: N=10 samples at temperature 0.7, majority vote ratio

Task 2.3  benchmark/baselines/attention_entropy.py
  - AttentionEntropyBaseline: reuse GradientAnchorFinder, return concentration score
```

### Phase 3: Metrics

```
Task 3.1  benchmark/metrics.py
  - compute_auroc, compute_ece, compute_brier, compute_risk_coverage, compute_accuracy_at_k
  - Use sklearn.metrics where available
```

### Phase 4: Runner and Report

```
Task 4.1  benchmark/runner.py
  - CLI: --dataset squad|funsd|cord|all, --method v1|v2|v3|logprob|sc|all, --n-examples N
  - Save per-example results to confidence_results_benchmark.json

Task 4.2  benchmark/report.py
  - Load JSON, print comparison table (AUROC + ECE)
  - Generate reliability diagram + risk-coverage curves
  - Display token attribution heatmaps for sampled examples
```

---

## 7. Implementation Architecture

The benchmark extends the existing repo structure. A new `benchmark/` directory is added alongside the existing `v1_basic/`, `v2_intermediate/`, `v3_advanced/`, and `shared/` directories.

```
benchmark/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── squad.py            # SQuAD 2.0 loader → list[BenchmarkExample]
│   ├── funsd.py            # FUNSD loader → list[BenchmarkExample]
│   └── cord.py             # CORD loader → list[BenchmarkExample]
├── baselines/
│   ├── __init__.py
│   ├── base.py             # Abstract BaseConfidenceMethod
│   ├── logprob.py          # Mean/Min/Norm token log-prob
│   ├── self_consistency.py # N=10 samples, majority vote
│   └── attention_entropy.py # Gradient attribution concentration
├── metrics.py              # AUROC, ECE, Brier, Risk-Coverage, Accuracy@K
├── visualize.py            # Token attribution heatmap (ANSI + matplotlib)
├── runner.py               # Run all methods × datasets, save JSON
└── report.py               # Load JSON, generate tables + figures
```

### Core Data Schemas

```python
@dataclass
class BenchmarkExample:
    id: str
    source_text: str
    query: str              # field name or question
    gold_answer: str        # normalized string (empty string if unanswerable)
    is_answerable: bool     # False for SQuAD unanswerable questions
    dataset: str            # 'squad', 'funsd', 'cord'

@dataclass
class MethodResult:
    example_id: str
    method: str
    confidence: float
    extracted_value: str
    is_correct: bool        # exact/fuzzy match against gold
    token_attributions: list | None   # from AnchorResult if available
```

---

## 8. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Perturbation method is expensive (K extra calls) | Evaluation takes long on CPU | Default K=5 for benchmarks; K sweep on SQuAD only |
| Gradient attribution requires full forward pass | Memory pressure on small GPU | Batch size 1; use CPU for gradient pass if needed |
| FUNSD/CORD OCR text is noisy | Fuzzy matching needed | Use character-level F1 matching in addition to exact match |
| SQuAD unanswerable questions: model may hallucinate | Confidence calibration affected | Report metrics separately for answerable vs unanswerable |
| Token attributions empty for v2/v3 | Anchor visualization fails | Add diagnostic check in runner; log warning |

---

## 9. Stretch Goals

1. **Combination experiment**: Show that Perturbation Confidence + Self-Consistency together beats either alone. This reframes the contribution from "alternative to" to "complementary with."
2. **Multi-field confidence**: Extend to documents with 10+ fields (e.g., CORD receipts). Compute document-level confidence as min(field confidences).
3. **Downstream task impact**: Use confidence scores to build a human-in-the-loop pipeline. Show that routing low-confidence extractions to human review yields 95%+ final accuracy with only 20% human review rate.

---

## 10. Target Venues

| Venue | Fit | Deadline Cycle |
|-------|-----|----------------|
| **EMNLP 2026** | Best fit — NLP + practical systems | ~June 2026 |
| **ACL 2026** | Strong fit — methodology + evaluation | ~January 2026 |
| **NeurIPS 2026** | If framed as general ML uncertainty | ~May 2026 |
| **COLM 2026** | New venue focused on LMs | TBD |
