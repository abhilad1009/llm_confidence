# Benchmarking the LLM Confidence Extractor: A Research Plan

## 1. Research Framing and Contribution

### Core Thesis

Existing LLM uncertainty methods rely on model internals (logprobs, embeddings) or stochastic sampling. This system introduces **perturbation-based confidence estimation**: instead of asking "how certain is the model?" it asks "does the model's behaviour change when we alter the source in controlled ways?" This is a fundamentally different and complementary paradigm — it measures *functional grounding* rather than *probabilistic belief*.

### Novelty Claims (What Reviewers Will Look For)

1. **Perturbation-based localization testing** — replacing the extracted value with type-matched random values to verify the model actually reads from the right location, not hallucinating or pattern-matching.
2. **Anchor sensitivity analysis** — destroying the surrounding context to test whether the model relies on genuine semantic understanding of the passage.
3. **Black-box compatibility** — unlike logprob or attention methods, Steps 3 and 4 work with *any* API-based LLM with zero access to internals.
4. **Composite scoring** that separates *where the model looked* from *whether the answer is right* — a distinction no existing method makes.

### Suggested Paper Title

> "Perturbation-Based Confidence Estimation for LLM Information Extraction: Measuring Functional Grounding Without Model Internals"

---

## 2. What to Benchmark Against (Baselines)

The system must be compared against every major family of LLM uncertainty estimation. Below are the six baselines, grouped by paradigm.

### 2.1 Token-Level Baselines (Require Logprob Access)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Mean Token Log-Probability** | Average log-prob of all output tokens | Simplest and most common LLM confidence proxy |
| **Min Token Log-Probability** | Lowest log-prob token in the answer | Captures "weakest link" uncertainty |
| **Normalized Sequence Probability** | Length-normalized product of all token probs | Controls for answer length bias |

**Implementation**: Call the LLM with `logprobs=True`, extract per-token log-probabilities, compute the three aggregations above.

### 2.2 Sampling-Based Baselines (Black-Box Compatible)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Self-Consistency (SC)** | Sample N outputs (temperature > 0), score = fraction agreeing with the majority answer | Gold-standard black-box method (Wang et al., 2023) |
| **Semantic Entropy** | Cluster sampled answers by semantic equivalence, compute entropy over clusters | State-of-the-art (Kuhn et al., 2023); handles paraphrase variation |

**Implementation**: For SC, sample 10–20 completions at temperature 0.7, normalize answers, compute majority agreement ratio. For Semantic Entropy, use a lightweight NLI model (e.g., DeBERTa-v3-large) to cluster semantically equivalent answers, then compute cluster-level entropy.

### 2.3 Prompting-Based Baselines (Zero Cost)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Verbalized Confidence** | Append "Rate your confidence 0–100" to the prompt | Tests whether models can self-assess |
| **P(True) Probing** | Ask "Is [extracted value] correct? Answer True/False" and use the logprob of "True" | Kadavath et al. (2022) showed this is surprisingly effective |

### 2.4 Attention-Based Baseline (Partial Overlap With Your System)

| Baseline | How It Works | Why Include It |
|----------|-------------|----------------|
| **Attention Entropy** | Compute entropy of the attention distribution over source tokens during generation | Isolates the contribution of your Step 2 (anchor finding) |

This is important because your system *includes* attention concentration as a component — you need to show the perturbation steps add value beyond attention alone.

---

## 3. What Datasets to Evaluate On

The benchmark suite must cover multiple extraction difficulty levels and domains.

### 3.1 Primary Evaluation Datasets

| Dataset | Domain | Task | Why It Matters |
|---------|--------|------|----------------|
| **SQuAD 2.0** | Wikipedia | Extractive QA + unanswerable | Has unanswerable questions → tests if confidence is low when model should abstain |
| **Natural Questions (NQ)** | Google queries | Open-domain QA | Real user questions; long documents; high ambiguity |
| **CUAD** | Legal contracts | Clause extraction | Professional IE with precise spans; high stakes |
| **FUNSD** | Scanned forms | Form key-value extraction | Noisy OCR + layout; tests robustness |
| **CORD** | Receipts | Receipt field extraction | Structured extraction with positional reasoning |

### 3.2 Stress-Test Datasets (Construct Yourself)

These controlled datasets isolate specific failure modes and are essential for publishable work.

| Stress Test | Construction Method | What It Tests |
|-------------|-------------------|---------------|
| **Duplicate Values** | Insert the same value (e.g., "$500") in multiple locations in the document | Can the system identify *which* "$500" the model used? |
| **Near-Miss Values** | Place similar but different values close together (e.g., "2023-01-15" and "2023-01-16") | Localization precision under ambiguity |
| **Adversarial Context** | Add misleading context near the correct answer (e.g., "The revenue was NOT $5M but rather...") | Does confidence drop for adversarial phrasing? |
| **Hallucination Trap** | Ask for values that don't exist in the document | Confidence should be near zero; tests calibration floor |
| **Long-Range Dependency** | Place the answer 2000+ tokens from the relevant question/header | Tests attention-based anchor finding at scale |
| **Cross-Language** | Source in one language, extraction prompt in another | Robustness of the perturbation pipeline |

### 3.3 Dataset Scale

For each dataset: sample **500–1000 examples** for main results, with a held-out **200-example dev set** for weight tuning (w₁, w₂, w₃).

---

## 4. Evaluation Metrics (How to Score a Confidence Method)

A confidence score is only useful if it *predicts whether the extraction is correct*. The ground-truth label for each example is binary: **correct extraction** (1) vs **incorrect extraction** (0), determined by exact match or fuzzy match against gold labels.

### 4.1 Discrimination Metrics

These measure whether high-confidence predictions are more likely to be correct.

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **AUROC** | Overall ranking quality — can the score separate correct from incorrect? | Higher is better; > 0.80 is strong |
| **AUPRC** | Precision-recall trade-off; better than AUROC for imbalanced data | Report for both correct and incorrect classes |
| **Spearman ρ** | Rank correlation between confidence and a continuous correctness measure (e.g., F1 per example) | Higher is better |

### 4.2 Calibration Metrics

These measure whether a confidence of X% means the model is correct X% of the time.

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Expected Calibration Error (ECE)** | Mean gap between confidence and actual accuracy in equal-width bins | Lower is better; < 0.05 is well-calibrated |
| **Maximum Calibration Error (MCE)** | Worst-case bin error | Lower is better |
| **Brier Score** | Mean squared error between confidence and binary correctness | Lower is better; combines calibration + discrimination |
| **Reliability Diagram** | Visual: plot confidence bins (x) vs actual accuracy (y) | Should follow the diagonal |

### 4.3 Selective Prediction Metrics

These simulate the real-world use case: "only trust extractions above threshold T."

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Risk-Coverage Curve** | At each coverage level (fraction of examples kept), what is the error rate? | Lower error at each coverage level |
| **AUC of Risk-Coverage** | Area under the above curve | Lower is better |
| **Accuracy@K% Coverage** | Accuracy when keeping only the top K% most confident predictions | Higher is better; report K = 90%, 80%, 70%, 50% |
| **Optimal F1 Threshold** | The threshold T that maximizes F1 on "accept if confident" decisions | Should be stable across datasets |

### 4.4 Efficiency Metric

Since the perturbation approach requires K additional LLM calls, cost matters.

| Metric | What It Measures |
|--------|-----------------|
| **Confidence AUROC per LLM Call** | AUROC achieved divided by total LLM calls used (1 + K_loc + K_anchor) |
| **Break-Even K** | Minimum K where perturbation methods outperform sampling-based methods at equal cost |

---

## 5. Experimental Design

### 5.1 Main Experiment Matrix

Run **every baseline × every dataset × 3 LLMs**:

**LLMs to test:**
- GPT-4o (closed-source, API-only — tests black-box path)
- Claude 3.5 Sonnet (closed-source, API-only)
- Llama 3.1 70B (open-weight — enables attention-based steps)

This gives you a **6 baselines × 5 datasets × 3 models = 90 cells** results table, plus your system.

### 5.2 Ablation Studies (Critical for Publication)

| Ablation | What You Remove | What It Shows |
|----------|----------------|---------------|
| **No attention** (w₁ = 0) | Remove attention concentration | Value of attention signals vs perturbation alone |
| **No localization** (w₂ = 0) | Remove Step 3 | Value of value-replacement testing |
| **No anchor validation** (w₃ = 0) | Remove Step 4 | Value of context destruction testing |
| **Localization only** | Only Step 3 | Standalone value of localization |
| **Anchor only** | Only Step 4 | Standalone value of anchor testing |
| **Perturbation only** (w₁ = 0) | Remove attention, keep Steps 3+4 | The pure black-box version |

### 5.3 Sensitivity Analysis

- **K (number of perturbation trials)**: Sweep K = {3, 5, 10, 15, 20}. Plot AUROC vs K. Find the elbow point.
- **Weight tuning**: Learn optimal (w₁, w₂, w₃) via logistic regression on the dev set; compare to default 0.2/0.4/0.4.
- **Temperature sensitivity**: Run Self-Consistency at temperatures {0.3, 0.5, 0.7, 1.0} to ensure fair comparison.
- **Random seed robustness**: Run the full pipeline 5 times with different random perturbation seeds; report variance.

### 5.4 Qualitative Analysis

Include 10–15 worked examples showing:
- Cases where perturbation confidence is low but logprob confidence is high (hallucinated but fluent)
- Cases where perturbation confidence is high but the answer is wrong (failure mode analysis)
- Reliability diagrams overlaid for all methods

---

## 6. Implementation Plan

### Phase 0: Infrastructure (Week 1–2)

```
Task 0.1  Set up evaluation harness
  - Standardized extraction prompt templates per dataset
  - Gold-label comparison function (exact match + fuzzy match + F1)
  - Logging: every LLM call, its input, output, latency, cost
  - Reproducibility: seed management, config files

Task 0.2  Implement all baselines
  - LogprobConfidence class (mean, min, length-normalized)
  - SelfConsistencyConfidence class (N=10 samples, majority vote)
  - SemanticEntropyConfidence class (DeBERTa-v3 clustering)
  - VerbalizedConfidence class (prompt-append method)
  - PTrueConfidence class (verification prompt + logprob)
  - AttentionEntropyConfidence class (for open-weight models)

Task 0.3  Implement your system
  - PerturbationConfidence class wrapping Steps 1–5
  - Type-aware random value generator (numbers, dates, names, currencies, ...)
  - Anchor context randomizer (preserve structure, replace content words)
  - Configurable K and weights
```

### Phase 1: Dataset Preparation (Week 2–3)

```
Task 1.1  Download and preprocess standard datasets
  - SQuAD 2.0: extract (passage, question, answer, is_answerable)
  - NQ: filter to short-answer subset
  - CUAD: map clause types to extraction targets
  - FUNSD / CORD: convert to (document_text, field_name, gold_value) triples

Task 1.2  Construct stress-test datasets
  - 100 examples each for: duplicate values, near-miss, adversarial context,
    hallucination trap, long-range dependency
  - Template-based generation for reproducibility
  - Manual quality check on 20% sample

Task 1.3  Dev/test splits
  - 200-example dev set per dataset for weight tuning
  - Remaining for test
  - Stratify by extraction difficulty (ensure mix of easy/hard)
```

### Phase 2: Run Main Experiments (Week 3–5)

```
Task 2.1  Run all baselines on all datasets (parallelize across models)
  - Log: (example_id, model, method, confidence_score, extracted_value, is_correct)
  - Estimated API cost: ~$200–500 depending on K and dataset sizes
  - For perturbation method with K=10: budget 11 calls per example

Task 2.2  Compute all metrics
  - AUROC, AUPRC, ECE, Brier, Risk-Coverage AUC per (method, dataset, model)
  - Statistical significance: paired bootstrap test (10,000 resamples)
    comparing your method's AUROC to each baseline

Task 2.3  Generate result tables and figures
  - Main table: methods (rows) × datasets (columns), AUROC values
  - Calibration figure: reliability diagrams for all methods on 2 datasets
  - Risk-Coverage figure: curves for all methods on 2 datasets
  - Cost-efficiency scatter: AUROC vs LLM calls
```

### Phase 3: Ablation and Analysis (Week 5–6)

```
Task 3.1  Run ablation experiments
  - All 6 ablation variants on all datasets

Task 3.2  K sensitivity sweep
  - K = {3, 5, 10, 15, 20} on 2 representative datasets

Task 3.3  Weight optimization
  - Logistic regression to learn (w₁, w₂, w₃) on dev set
  - Compare learned vs default weights on test set

Task 3.4  Qualitative error analysis
  - Sample 50 examples where your method and best baseline disagree most
  - Categorize failure modes
  - Select 10–15 for paper figures
```

### Phase 4: Paper Writing (Week 6–8)

```
Task 4.1  Draft paper structure (see Section 7 below)
Task 4.2  Write experiments and results first (most reviewers read this first)
Task 4.3  Write method section with clear algorithmic description
Task 4.4  Write introduction last (needs to frame contribution precisely)
Task 4.5  Internal review cycle (2 rounds)
```

---

## 7. Paper Structure (Target: ACL / EMNLP / NeurIPS)

```
1. Introduction                                         (~1 page)
   - Problem: LLMs extract information but give no confidence signal
   - Gap: existing uncertainty methods don't test functional grounding
   - Contribution: perturbation-based confidence, black-box compatible, 
     outperforms baselines on discrimination and calibration

2. Related Work                                         (~1 page)
   - LLM uncertainty estimation (logprobs, sampling, verbalized)
   - Calibration of language models
   - Information extraction with LLMs
   - Perturbation methods in NLP (input perturbation for robustness,
     not for confidence — key distinction)

3. Method                                               (~2 pages)
   - Problem formulation
   - Step-by-step algorithm with pseudocode
   - Type-aware perturbation generation
   - Composite scoring and weight interpretation
   - Complexity analysis: O(K) additional LLM calls

4. Experimental Setup                                   (~1.5 pages)
   - Datasets (standard + stress-test)
   - Baselines
   - Models
   - Metrics
   - Implementation details (K=10, temperature, prompt templates)

5. Results                                              (~2.5 pages)
   - Main comparison table (AUROC, ECE, Risk-Coverage AUC)
   - Calibration analysis (reliability diagrams)
   - Selective prediction analysis (risk-coverage curves)
   - Cost-efficiency analysis

6. Analysis                                             (~1.5 pages)
   - Ablation study
   - K sensitivity
   - Weight analysis
   - Qualitative examples and failure modes
   - When does the method fail? (honest limitations)

7. Conclusion                                           (~0.5 page)

Appendix
   - Full prompt templates
   - All per-dataset results
   - Stress-test dataset construction details
   - Statistical significance tables
```

---

## 8. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Perturbation method is expensive (K extra calls) | Reviewers dismiss on cost | Show cost-efficiency plot; demonstrate K=5 is sufficient; frame as offline quality assurance, not real-time |
| Self-Consistency outperforms at equal cost | Undermines contribution | Focus on the *complementary* information: perturbation tests localization, not just consistency. Show combining both yields best results |
| Attention step doesn't work for API models | Limits applicability | Present "black-box variant" (w₁=0) as the primary system; attention is a bonus for open models |
| Type-aware perturbation is fragile | Breaks on unusual types | Include a "generic" fallback (random string replacement); report robustness across types |
| Weight tuning overfits to dev set | Generalization concerns | Cross-validate across datasets; show default weights are competitive |

---

## 9. Stretch Goals for Stronger Publication

1. **Combination experiment**: Show that Perturbation Confidence + Self-Consistency together (as a two-feature confidence) beats either alone. This reframes the contribution from "alternative to" to "complementary with."

2. **Downstream task impact**: Use confidence scores to build a human-in-the-loop pipeline. Show that routing low-confidence extractions to human review yields 95%+ final accuracy with only 20% human review rate. This makes the paper immediately practical.

3. **Multi-field confidence**: Extend to documents with 10+ fields (e.g., invoices). Show how per-field confidence enables field-level quality control. Compute document-level confidence as min(field confidences).

4. **Theoretical analysis**: Provide a probabilistic interpretation of the perturbation test as estimating a lower bound on P(model uses correct source location | extraction).

---

## 10. Implementation Architecture

```
confidence_benchmark/
├── configs/
│   ├── datasets.yaml          # Dataset paths, splits, field definitions
│   ├── models.yaml            # LLM configs (API keys, endpoints, params)
│   └── experiments.yaml       # Method configs, K values, weight grids
│
├── data/
│   ├── loaders/
│   │   ├── squad.py
│   │   ├── natural_questions.py
│   │   ├── cuad.py
│   │   ├── funsd.py
│   │   └── cord.py
│   ├── stress_tests/
│   │   ├── generator.py       # Template-based stress test generation
│   │   └── templates/         # Per-stress-test templates
│   └── preprocessing.py       # Normalize to (source, query, gold_answer)
│
├── methods/
│   ├── base.py                # Abstract ConfidenceMethod interface
│   ├── logprob.py             # Mean/Min/Normalized token logprob
│   ├── self_consistency.py    # Sample N, majority vote ratio
│   ├── semantic_entropy.py    # NLI-based clustering + entropy
│   ├── verbalized.py          # Prompt-appended self-assessment
│   ├── p_true.py              # Verification prompt + logprob
│   ├── attention_entropy.py   # For open-weight models
│   └── perturbation.py        # YOUR SYSTEM (Steps 1-5)
│       ├── extractor.py       # Step 1: Base extraction
│       ├── anchor_finder.py   # Step 2: Attention or fallback
│       ├── localizer.py       # Step 3: Value replacement test
│       ├── anchor_validator.py # Step 4: Context destruction test
│       ├── scorer.py          # Step 5: Weighted combination
│       └── type_perturber.py  # Type-aware random value generator
│
├── evaluation/
│   ├── metrics.py             # AUROC, AUPRC, ECE, Brier, Risk-Coverage
│   ├── calibration.py         # Reliability diagrams, binning
│   ├── selective.py           # Risk-coverage curves, Accuracy@K
│   ├── significance.py        # Paired bootstrap tests
│   └── cost.py                # LLM call counting, cost-efficiency
│
├── experiments/
│   ├── run_main.py            # Main comparison: all methods × datasets × models
│   ├── run_ablation.py        # Ablation variants
│   ├── run_sensitivity.py     # K sweep, temperature sweep
│   ├── run_weight_tuning.py   # Logistic regression weight learning
│   └── run_stress_test.py     # Stress-test suite
│
├── analysis/
│   ├── tables.py              # Generate LaTeX tables
│   ├── figures.py             # Generate matplotlib/seaborn figures
│   └── qualitative.py         # Sample disagreement examples
│
├── scripts/
│   ├── download_datasets.sh
│   └── estimate_cost.py       # Pre-flight cost estimation
│
└── tests/
    ├── test_perturbation.py   # Unit tests for type-aware perturbation
    ├── test_metrics.py        # Metric computation correctness
    └── test_integration.py    # End-to-end on tiny dataset
```

---

## 11. Estimated Timeline and Budget

| Phase | Duration | API Cost Estimate |
|-------|----------|-------------------|
| Phase 0: Infrastructure | 2 weeks | $0 (local dev) |
| Phase 1: Data Preparation | 1 week | $0 |
| Phase 2: Main Experiments | 2 weeks | $300–600 |
| Phase 3: Ablation + Analysis | 1.5 weeks | $150–300 |
| Phase 4: Paper Writing | 2 weeks | $0 |
| **Total** | **~8.5 weeks** | **~$450–900** |

API cost breakdown (Phase 2): 5 datasets × 800 examples × (6 baselines needing ~15 calls + perturbation needing ~11 calls) ≈ 400,000 LLM calls. At ~$0.001/call average, that's ~$400.

---

## 12. Target Venues

| Venue | Fit | Deadline Cycle |
|-------|-----|----------------|
| **EMNLP 2026** | Best fit — NLP + practical systems | ~June 2026 |
| **ACL 2026** | Strong fit — methodology + evaluation | ~January 2026 |
| **NeurIPS 2026** | If framed as general ML uncertainty | ~May 2026 |
| **NAACL 2026** | Good fit, slightly lower bar | ~October 2025 |
| **COLM 2026** | New venue focused on LMs | TBD |

The EMNLP framing would be: "Perturbation-Based Confidence for LLM Information Extraction" under the Information Extraction or NLP Applications track.