"""
shared/models.py — HuggingFace model wrapper used by all three pipeline versions.

v1_basic uses only .extract()
v2_intermediate also uses .forward_with_grad()
v3_advanced subclasses this in v3_advanced/models.py to add logit capture + MC dropout

Default model: Qwen/Qwen3-0.6B
  • Chat template via tokenizer.apply_chat_template()
  • Optional 4-bit NF4 quantization (bitsandbytes)
  • Supports seq2seq (T5/FLAN-T5) and causal LM (Qwen, Llama-3, Mistral, Phi-3)

4-bit quantization note
-----------------------
For the gradient attribution pass we dequantize the embedding layer to fp32
before the backward call — adds ~40 ms but keeps VRAM flat.
"""

from __future__ import annotations

import os
os.environ['HF_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import re
import sys
import traceback
import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)

try:
    import bitsandbytes  # noqa: F401
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


def _bnb_cuda_ok() -> bool:
    try:
        import bitsandbytes as bnb
        x = torch.ones(1, 64, device="cuda", dtype=torch.float16)
        bnb.nn.Linear8bitLt(64, 64)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SEQ2SEQ_TEMPLATE = (
    "Extract the value of '{field}' from the text below. "
    "Reply with only the extracted value and nothing else.\n\n"
    "Text: {text}\n\n"
    "{field}:"
)

CAUSAL_TEMPLATE = (
    "### Instruction\n"
    "Extract the value of '{field}' from the text below. "
    "Reply with only the extracted value and nothing else.\n\n"
    "### Text\n{text}\n\n"
    "### Answer\n"
    "{field}: "
)

_CHAT_SYSTEM = (
    "You are a precise information-extraction assistant. "
    "When asked to extract a field, reply with only the extracted value — "
    "no explanation, no punctuation, no extra words."
)


def _build_chat_prompt(tokenizer, text: str, field: str) -> str:
    user_msg = (
        f"Extract the value of '{field}' from the text below. "
        f"Reply with only the extracted value and nothing else.\n\n"
        f"Text:\n{text}"
    )
    messages = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return CAUSAL_TEMPLATE.format(text=text, field=field)


def _clean(text: str) -> str:
    text = text.strip()
    if ":" in text:
        parts = text.split(":", 1)
        if len(parts[0].split()) <= 3:
            text = parts[1].strip()
    return text.split("\n")[0].strip()


def _make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HFExtractionModel:
    """
    Wraps a HuggingFace model for extraction + gradient attribution.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: 'Qwen/Qwen3-0.6B'
    device : str, optional
        'cpu', 'cuda', or 'mps'. Auto-detected if None.
    load_in_4bit : bool, optional
        Load in 4-bit NF4 (requires bitsandbytes).
    max_input_tokens : int
        Prompt truncation limit.
    max_new_tokens : int
        Max tokens generated for the extracted value.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        bnb_ok = _BNB_AVAILABLE and self.device == "cuda" and _bnb_cuda_ok()
        use_4bit = load_in_4bit if load_in_4bit is not None else bnb_ok
        if load_in_4bit and not bnb_ok:
            print("[HFExtractionModel] WARNING: 4-bit requested but bitsandbytes "
                  "CUDA check failed — falling back to fp16.")
            use_4bit = False

        self._quantized = use_4bit
        quant_label = "4-bit NF4" if use_4bit else "fp16"
        print(f"[HFExtractionModel] Loading '{model_name}' ({quant_label}) on {self.device} …")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        load_kwargs: dict = {"trust_remote_code": True}
        if use_4bit:
            load_kwargs["quantization_config"] = _make_bnb_config()
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = (
                torch.float16 if self.device == "cuda" else torch.float32
            )
            if self.device == "cuda":
                load_kwargs["device_map"] = {"": "cuda:0"}

        _CAUSAL_ONLY = ("qwen", "llama", "mistral", "phi", "gemma", "gpt")
        force_causal = any(k in model_name.lower() for k in _CAUSAL_ONLY)

        def _log(msg):
            print(msg, flush=True)

        if not force_causal:
            try:
                _log("[HFExtractionModel] Trying AutoModelForSeq2SeqLM …")
                self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, **load_kwargs
                )
                self.is_seq2seq = True
                _log("[HFExtractionModel] Loaded as seq2seq.")
            except Exception as e1:
                _log(f"[HFExtractionModel] Seq2seq failed ({type(e1).__name__}: {e1}) — trying CausalLM …")
                force_causal = True

        if force_causal:
            try:
                _log("[HFExtractionModel] Step 1/3: Initialising model config …")
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                _log(f"[HFExtractionModel] Step 1/3: Config OK — {cfg.__class__.__name__}")

                # Load without device_map so weights land in CPU RAM as plain tensors,
                # then .to(self.device) correctly moves them to CUDA.
                # (device_map="cpu" creates a dispatch model that ignores .to())
                cpu_kwargs = {k: v for k, v in load_kwargs.items() if k != "device_map"}
                cpu_kwargs["low_cpu_mem_usage"] = True

                _log("[HFExtractionModel] Step 2/3: Loading weights to CPU RAM …")
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **cpu_kwargs)
                self.is_seq2seq = False
                _log("[HFExtractionModel] Step 2/3: Weights loaded to CPU OK.")

                _log(f"[HFExtractionModel] Step 3/3: Moving model to {self.device} …")
                self.model = self.model.to(self.device)
                _log(f"[HFExtractionModel] Step 3/3: Model on {self.device} OK.")

            except BaseException as exc:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load '{model_name}'.\n"
                    f"Error: {type(exc).__name__}: {exc}\n\n"
                    "Common fixes:\n"
                    "  1. Increase Windows page file to 16 GB+\n"
                    "  2. uv pip install bitsandbytes --pre then set load_in_4bit=True\n"
                    "  3. Verify CUDA: python -c \"import torch; print(torch.cuda.is_available())\"\n"
                ) from exc

        if not use_4bit and self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self._model_dtype = next(self.model.parameters()).dtype

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._use_chat_template = (
            not self.is_seq2seq
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        )

        print(f"[HFExtractionModel] Ready  "
              f"(seq2seq={self.is_seq2seq}, "
              f"chat_template={self._use_chat_template}, "
              f"4bit={self._quantized})")

    # ── Extraction ─────────────────────────────────────────────────────────────

    def build_prompt(self, text: str, field: str) -> str:
        if self.is_seq2seq:
            return SEQ2SEQ_TEMPLATE.format(text=text, field=field)
        if self._use_chat_template:
            return _build_chat_prompt(self.tokenizer, text, field)
        return CAUSAL_TEMPLATE.format(text=text, field=field)

    @torch.no_grad()
    def extract(self, text: str, field: str) -> str:
        """Run greedy extraction and return cleaned value string."""
        prompt = self.build_prompt(text, field)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
        ).to(self.device)

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

    # ── Gradient Attribution ────────────────────────────────────────────────────

    def forward_with_grad(
        self,
        text: str,
        field: str,
        target_value: str,
    ) -> tuple[torch.Tensor, list]:
        """
        Differentiable forward pass for token-level attribution.

        Returns
        -------
        attributions : Tensor (seq_len,)
            Per-input-token attribution score  ||grad · emb||₂
        offset_mapping : list[(int,int)]
            Character offsets for each input token in the prompt.
        """
        prompt = self.build_prompt(text, field)

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        offset_mapping = enc["offset_mapping"][0].tolist()

        tgt_enc = self.tokenizer(target_value, return_tensors="pt")
        labels = tgt_enc["input_ids"].to(self.device)

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            raw_embeds = embed_layer(input_ids).to(self._model_dtype)
        embeds = raw_embeds.detach().requires_grad_(True)

        if self.is_seq2seq:
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            with torch.no_grad():
                tgt_embeds = embed_layer(labels).to(self._model_dtype).detach()
            combined_embeds = torch.cat([embeds, tgt_embeds], dim=1)
            ignore = torch.full(
                (1, embeds.shape[1]), -100, dtype=torch.long, device=self.device
            )
            combined_labels = torch.cat([ignore, labels], dim=1)
            outputs = self.model(
                inputs_embeds=combined_embeds,
                labels=combined_labels,
            )

        outputs.loss.backward()

        grads = embeds.grad
        attr = (grads * embeds.detach()).norm(dim=-1).squeeze(0)
        attr = attr / (attr.max() + 1e-9)

        return attr.detach().cpu(), offset_mapping

    # ── Vocabulary helpers ──────────────────────────────────────────────────────

    @property
    def vocab_words(self) -> list[str]:
        """Return plausible English-looking vocabulary tokens for corruption."""
        if not hasattr(self, "_vocab_cache"):
            vocab = self.tokenizer.get_vocab()
            self._vocab_cache = [
                t.lstrip("▁Ġ")
                for t in vocab
                if re.match(r"^[▁Ġ]?[a-zA-Z]{4,}$", t)
            ][:5000]
        return self._vocab_cache
