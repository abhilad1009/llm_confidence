"""
v3_advanced/mlm_scorer.py — Masked Language Model anchor perplexity scorer.

Intuition
---------
A well-grounded anchor extracted from a real document should read like
natural text — a language model trained on natural text should assign it a
relatively low perplexity.

Artificially injected, paraphrased, or hallucinated anchor text tends to be
less natural and scores higher perplexity.

We estimate the *pseudo-log-likelihood* (PLL) of the anchor text using a
small BERT-style MLM (default: distilbert-base-uncased):

  For each token position i:
    1. Mask token i.
    2. Run the MLM.
    3. Record  log P(token_i | all other tokens).
  PLL = Σ log P(token_i | context_i)
  PPL = exp( −PLL / n_tokens )

Interpretation
--------------
  Low PPL   →  anchor is natural / coherent            →  higher confidence
  High PPL  →  anchor is unusual or incoherent         →  lower confidence

The raw PPL value is normalised against a rough empirical baseline so the
final mlm_score lies in [0, 1]:
  mlm_score = 1 / (1 + log(ppl / baseline_ppl))
where baseline_ppl ≈ 50 for distilbert-base-uncased on general English text.
"""

from __future__ import annotations

import os
os.environ['HF_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import math
import torch
from typing import Optional


_DEFAULT_MODEL = "distilbert-base-uncased"
_BASELINE_PPL = 50.0   # rough empirical baseline for distilbert on clean English


class MLMAnchorScorer:
    """
    Computes pseudo-perplexity of anchor text using a masked language model.

    Parameters
    ----------
    model_name : str
        HuggingFace MLM model ID.  Default: 'distilbert-base-uncased'
    device : str, optional
        'cpu', 'cuda', or 'mps'.  Auto-detected if None.
    max_length : int
        Maximum token length to score (long anchors are truncated).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy load the MLM (only when first used)."""
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        print(f"[MLMAnchorScorer] Loading '{self.model_name}' …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print(f"[MLMAnchorScorer] Ready on {self.device}.")

    def compute_perplexity(self, text: str) -> float:
        """
        Compute pseudo-perplexity of *text* using masked token scoring.

        Returns
        -------
        float
            Pseudo-perplexity value.  Typical range: 5–200 for English.
        """
        self._load()

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)          # (1, seq_len)
        n_tokens = input_ids.shape[1]

        if n_tokens <= 2:  # only [CLS] and [SEP]
            return _BASELINE_PPL

        mask_token_id = self._tokenizer.mask_token_id
        log_prob_sum = 0.0
        scored_tokens = 0

        # Score each non-special token by masking it one at a time
        for i in range(1, n_tokens - 1):  # skip [CLS] and [SEP]
            masked_ids = input_ids.clone()
            original_token_id = int(masked_ids[0, i].item())
            masked_ids[0, i] = mask_token_id

            with torch.no_grad():
                logits = self._model(masked_ids).logits   # (1, seq_len, vocab)

            token_logits = logits[0, i]                   # (vocab,)
            log_probs = torch.log_softmax(token_logits, dim=-1)
            log_prob_sum += log_probs[original_token_id].item()
            scored_tokens += 1

        if scored_tokens == 0:
            return _BASELINE_PPL

        avg_neg_log_prob = -log_prob_sum / scored_tokens
        ppl = math.exp(avg_neg_log_prob)
        return ppl

    def compute_mlm_score(self, text: str) -> float:
        """
        Compute a normalised MLM confidence score in [0, 1].

        Higher score = lower perplexity = more natural / grounded anchor.

        Returns
        -------
        float
            mlm_score ∈ [0, 1].
        """
        ppl = self.compute_perplexity(text)
        # Sigmoid-like normalisation anchored at baseline_ppl
        # score = 1 at ppl=1, 0.5 at ppl=baseline, approaches 0 for very high ppl
        score = 1.0 / (1.0 + math.log(max(ppl, 1.0) / _BASELINE_PPL + 1e-9))
        return max(0.0, min(1.0, score))
