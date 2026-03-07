"""
v3_advanced/models.py — Extended HF model with logit capture and MC dropout.

Extends HFExtractionModel from shared/models.py with:

  extract_with_logits(text, field)
      Runs greedy generation with output_scores=True, returning both the
      extracted value string and the per-token logit tensors.  These logits
      are passed to entropy_scorer to compute generation entropy.

  extract_mc(text, field, n_passes, dropout_p)
      Monte Carlo dropout: temporarily enables dropout during inference and
      runs n_passes forward passes.  The spread of extracted values is a
      proxy for model uncertainty.  Fully implemented but not yet wired into
      the main pipeline confidence score (stub for further research).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from shared.models import HFExtractionModel, _clean


class AdvancedHFExtractionModel(HFExtractionModel):
    """
    HFExtractionModel extended with logit capture and MC dropout.

    All parameters are identical to HFExtractionModel.
    """

    # ── Logit-capturing extraction ─────────────────────────────────────────────

    @torch.no_grad()
    def extract_with_logits(
        self, text: str, field: str
    ) -> tuple[str, list[torch.Tensor]]:
        """
        Run greedy extraction and return (value_string, per_token_logits).

        Returns
        -------
        value : str
            Cleaned extracted value.
        logits_list : list[Tensor]
            One (vocab_size,) tensor per generated token.
            Empty list for seq2seq models (logits not easily accessible).
        """
        prompt = self.build_prompt(text, field)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
        ).to(self.device)

        if self.is_seq2seq:
            # Seq2seq — standard generate, no per-step logits collected
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return _clean(decoded), []

        # Causal LM — capture output_scores (one tensor per generated token)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        new_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # out.scores is a tuple of (vocab_size,) tensors, one per generated step
        logits_list: list[torch.Tensor] = [s.squeeze(0).cpu() for s in out.scores]

        return _clean(decoded), logits_list

    # ── Monte Carlo dropout ────────────────────────────────────────────────────

    def extract_mc(
        self,
        text: str,
        field: str,
        n_passes: int = 20,
        dropout_p: float = 0.1,
    ) -> list[str]:
        """
        Monte Carlo dropout: run n_passes with dropout enabled.

        Returns a list of extracted value strings.  High variance in the list
        indicates high model uncertainty.

        Parameters
        ----------
        n_passes : int
            Number of stochastic forward passes.
        dropout_p : float
            Dropout probability applied to all dropout layers.

        Notes
        -----
        Temporarily patches all nn.Dropout layers to the given probability and
        puts the model in .train() mode (which activates dropout) for the
        duration of the passes, then restores the original state.
        """
        # Patch dropout layers
        original_ps: dict[nn.Dropout, float] = {}
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                original_ps[m] = m.p
                m.p = dropout_p

        self.model.train()
        results: list[str] = []
        try:
            for _ in range(n_passes):
                try:
                    val = self._extract_train_mode(text, field)
                    results.append(val)
                except Exception:
                    pass
        finally:
            self.model.eval()
            for m, p in original_ps.items():
                m.p = p

        return results

    def _extract_train_mode(self, text: str, field: str) -> str:
        """Single extraction pass in train mode (dropout active), with gradients disabled."""
        prompt = self.build_prompt(text, field)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            if self.is_seq2seq:
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
                decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            else:
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _clean(decoded)
