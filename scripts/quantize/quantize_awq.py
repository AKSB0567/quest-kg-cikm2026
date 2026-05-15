"""4-bit AWQ quantization for 7-8B LLMs to fit Colab T4/V100 (16GB) and 1080 Ti (11GB).

Strategy: for each backbone, either (a) download a pre-quantized AWQ from HF if
available, or (b) quantize ourselves with AutoAWQ using a calibration set.

Pre-quantized HF mirrors (preferred — saves hours):
    - Mistral-7B-Instruct-v0.3 -> solidrust/Mistral-7B-Instruct-v0.3-AWQ
    - LLaMA-3.1-8B-Instruct    -> casperhansen/llama-3-8b-instruct-awq (or hugging-quants variants)
    - Qwen2.5-7B-Instruct      -> Qwen/Qwen2.5-7B-Instruct-AWQ (official, if available)
    - LLaMA-3.2-3B-Instruct    -> SmallDoge/Llama-3.2-3B-Instruct-AWQ (community)

If a pre-quantized variant isn't available for a model, this script will
quantize from the fp16 base using a small calibration dataset.

Usage:
    python -m scripts.quantize.quantize_awq \
        --root /content/drive/MyDrive/quest_kg/models \
        --models llama32-3b mistral-7b qwen25-7b llama31-8b
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path


# Map our shortnames to (HF base id, pre-quantized AWQ id or None)
MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "llama32-3b":  ("meta-llama/Llama-3.2-3B-Instruct",     "hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4"),
    "mistral-7b":  ("mistralai/Mistral-7B-Instruct-v0.3",   "solidrust/Mistral-7B-Instruct-v0.3-AWQ"),
    "qwen25-7b":   ("Qwen/Qwen2.5-7B-Instruct",             "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    "llama31-8b":  ("meta-llama/Meta-Llama-3.1-8B-Instruct", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
}


def have_marker(out_dir: Path) -> bool:
    return (out_dir / ".done").exists()


def fetch_prequantized(hf_id: str, out_dir: Path) -> None:
    """Snapshot-download a pre-quantized AWQ model from HF Hub."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5"],
    )


def quantize_from_base(base_id: str, out_dir: Path) -> None:
    """Quantize fp16 base model to 4-bit AWQ with AutoAWQ using a small calib set."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"  loading base {base_id} (fp16)")
    model = AutoAWQForCausalLM.from_pretrained(base_id, safetensors=True, device_map="auto")
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

    # Tiny calibration set (~128 examples is enough for AWQ)
    calib = [
        "The mitochondrion is the powerhouse of the cell.",
        "What is the capital of France? The capital of France is Paris.",
        "In computer science, a knowledge graph represents entities and their relations.",
        "Explain the difference between supervised and unsupervised learning.",
    ] * 32

    print("  running AWQ quantization (this takes ~10-20 minutes on A100 / 30-60 minutes on T4)")
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))


def process_one(shortname: str, root: Path) -> None:
    base_id, awq_id = MODEL_REGISTRY[shortname]
    out_dir = root / shortname
    out_dir.mkdir(parents=True, exist_ok=True)
    if have_marker(out_dir):
        print(f"[{shortname}] already present at {out_dir}; skipping.")
        return

    print(f"[{shortname}] target: {out_dir}")
    if awq_id:
        try:
            print(f"  trying pre-quantized: {awq_id}")
            fetch_prequantized(awq_id, out_dir)
            (out_dir / ".done").write_text(f"prequant:{awq_id}\n")
            print(f"[{shortname}] OK (pre-quantized from {awq_id})")
            return
        except Exception as e:
            print(f"  pre-quantized fetch failed: {e}; falling back to local AWQ quantization.")

    try:
        quantize_from_base(base_id, out_dir)
        (out_dir / ".done").write_text(f"local-awq:{base_id}\n")
        print(f"[{shortname}] OK (locally quantized from {base_id})")
    except Exception as e:
        print(f"[{shortname}] FAILED: {e}")
        raise


def main(root: str, models: list[str]) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"unknown model shortname: {m}; skipping")
            continue
        process_one(m, root_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. /content/drive/MyDrive/quest_kg/models")
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()
    main(args.root, args.models)
