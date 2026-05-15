"""4-bit AWQ quantization for 7-8B LLMs to fit Colab T4/V100 (16GB) and 1080 Ti (11GB).

Strategy: for each backbone, try (a) pre-quantized AWQ from HF if available,
otherwise (b) quantize ourselves with AutoAWQ.

Default backbones are **NON-GATED** so they work without HF access requests:
  - qwen25-3b   : Qwen/Qwen2.5-3B-Instruct (small open)
  - mistral-7b  : mistralai/Mistral-7B-Instruct-v0.2 (medium open, different family)
  - qwen25-7b   : Qwen/Qwen2.5-7B-Instruct + pre-quant Qwen/Qwen2.5-7B-Instruct-AWQ

Optional GATED backbone (only attempted if --models includes 'llama31-8b'):
  - llama31-8b  : meta-llama/Meta-Llama-3.1-8B-Instruct (request access on HF)

If a model is gated and you lack access, the script skips it gracefully and
continues with the rest.

Usage:
    # default (non-gated only):
    python -m scripts.quantize.quantize_awq --root /content/drive/MyDrive/quest_kg/models

    # explicit list including gated LLaMA (only if you have access):
    python -m scripts.quantize.quantize_awq \\
        --root /content/drive/MyDrive/quest_kg/models \\
        --models qwen25-3b mistral-7b qwen25-7b llama31-8b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Map shortnames to (HF base id, pre-quantized AWQ id or None, is_gated)
MODEL_REGISTRY: dict[str, tuple[str, str | None, bool]] = {
    # Non-gated (default)
    "qwen25-3b":   ("Qwen/Qwen2.5-3B-Instruct",            "Qwen/Qwen2.5-3B-Instruct-AWQ",           False),
    "mistral-7b":  ("mistralai/Mistral-7B-Instruct-v0.2",  "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",  False),
    "qwen25-7b":   ("Qwen/Qwen2.5-7B-Instruct",            "Qwen/Qwen2.5-7B-Instruct-AWQ",           False),

    # Gated (optional; requires HF access request approved)
    "llama31-8b":  ("meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    True),

    # Legacy alias for backward-compat
    "llama32-3b":  ("meta-llama/Llama-3.2-3B-Instruct",    None,                                     True),
}

DEFAULT_MODELS = ["qwen25-3b", "mistral-7b", "qwen25-7b"]


def have_marker(out_dir: Path) -> bool:
    return (out_dir / ".done").exists()


def fetch_prequantized(hf_id: str, out_dir: Path) -> None:
    """Snapshot-download a pre-quantized AWQ model from HF Hub."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(out_dir),
        ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5"],
    )


def quantize_from_base(base_id: str, out_dir: Path) -> None:
    """Quantize fp16 base model to 4-bit AWQ with AutoAWQ + small calib set."""
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

    calib = [
        "The mitochondrion is the powerhouse of the cell.",
        "What is the capital of France? The capital of France is Paris.",
        "In computer science, a knowledge graph represents entities and their relations.",
        "Explain the difference between supervised and unsupervised learning.",
    ] * 32

    print("  running AWQ quantization (~10-60 min depending on GPU)")
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))


def is_gated_access_error(err: Exception) -> bool:
    s = str(err).lower()
    return any(k in s for k in ["gated", "403", "access to model", "restricted", "authorized"])


def process_one(shortname: str, root: Path) -> str:
    """Returns 'ok', 'skipped_gated', or 'failed'."""
    base_id, awq_id, gated = MODEL_REGISTRY[shortname]
    out_dir = root / shortname
    out_dir.mkdir(parents=True, exist_ok=True)
    if have_marker(out_dir):
        print(f"[{shortname}] already present at {out_dir}; skipping.")
        return "ok"

    print(f"[{shortname}] target: {out_dir} (gated={gated})")

    # Try pre-quantized first
    if awq_id:
        try:
            print(f"  trying pre-quantized: {awq_id}")
            fetch_prequantized(awq_id, out_dir)
            (out_dir / ".done").write_text(f"prequant:{awq_id}\n")
            print(f"[{shortname}] OK (pre-quantized from {awq_id})")
            return "ok"
        except Exception as e:
            if is_gated_access_error(e):
                print(f"[{shortname}] pre-quantized is gated and you lack access: {e}")
                # fall through to base attempt; usually also gated
            else:
                print(f"  pre-quantized fetch failed: {e}; falling back to local AWQ quantization.")

    # Try local quantization from base
    try:
        quantize_from_base(base_id, out_dir)
        (out_dir / ".done").write_text(f"local-awq:{base_id}\n")
        print(f"[{shortname}] OK (locally quantized from {base_id})")
        return "ok"
    except Exception as e:
        if is_gated_access_error(e):
            print(f"[{shortname}] SKIPPED: gated repo without access. "
                  f"Visit https://huggingface.co/{base_id} and click 'Request access', then rerun.")
            return "skipped_gated"
        print(f"[{shortname}] FAILED: {e}")
        return "failed"


def main(root: str, models: list[str]) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    summary = {"ok": [], "skipped_gated": [], "failed": [], "unknown": []}
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"unknown model shortname: {m}; available={list(MODEL_REGISTRY)}")
            summary["unknown"].append(m)
            continue
        result = process_one(m, root_path)
        summary[result].append(m)

    print("=" * 60)
    print("QUANTIZE SUMMARY")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60)
    if summary["skipped_gated"]:
        print("\nGated models skipped. To enable them:")
        for m in summary["skipped_gated"]:
            base_id, _, _ = MODEL_REGISTRY[m]
            print(f"  1. Visit https://huggingface.co/{base_id}")
            print(f"  2. Click 'Request access' (usually auto-approved within minutes).")
            print(f"  3. Re-run: python -m scripts.quantize.quantize_awq --root <root> --models {m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. /content/drive/MyDrive/quest_kg/models")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"default: {DEFAULT_MODELS}; add 'llama31-8b' if you have HF access")
    args = ap.parse_args()
    main(args.root, args.models)
