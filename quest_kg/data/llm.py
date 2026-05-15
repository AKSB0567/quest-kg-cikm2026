"""Unified LLM interface used by all baselines + QUEST-KG's answer-generation step.

Backend: HuggingFace Transformers loading 4-bit AWQ models from
`/content/drive/MyDrive/quest_kg/models/<shortname>/`.

Same model + same generation parameters across all methods -> matched-condition guarantee.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


class LLMInterface:
    """Wraps HF Transformers AWQ models with batched generation."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        if not HAS_TORCH:
            raise ImportError("torch is required to use LLMInterface")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        if not prompts:
            return []
        max_new = max_new_tokens or self.max_new_tokens
        out: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=(self.temperature > 0.0),
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Strip prompt tokens
            prompt_len = enc.input_ids.shape[1]
            for j, seq in enumerate(gen):
                gen_only = seq[prompt_len:]
                out.append(self.tokenizer.decode(gen_only, skip_special_tokens=True).strip())
        return out


def llm_path_for(shortname: str, drive_root: str = "/content/drive/MyDrive/quest_kg") -> Path:
    p = Path(drive_root) / "models" / shortname
    if not (p / ".done").exists():
        raise FileNotFoundError(f"{p} not present or quantization incomplete")
    return p
