# Model quantization

This script populates the Drive folder `/MyDrive/quest_kg/models/` with 4-bit AWQ
versions of every backbone used in QUEST-KG experiments. After this runs, every
notebook can simply `from_pretrained("/content/drive/MyDrive/quest_kg/models/mistral-7b")`
and load the quantized weights — fast, fits Colab T4/V100, and consistent across sessions.

## Why pre-quantize once?

- Re-downloading + re-quantizing in every Colab session burns hours.
- 4-bit AWQ shrinks a 7B model from ~13 GB (fp16) to ~4 GB — fits 1080 Ti and T4.
- Mistral-7B and Qwen-2.5-7B have public AWQ mirrors; we use those when possible.

## What gets saved

```
/content/drive/MyDrive/quest_kg/models/
├── llama32-3b/        ~2  GB  (Llama-3.2-3B-Instruct AWQ)
├── mistral-7b/        ~4  GB  (Mistral-7B-Instruct-v0.3 AWQ)
├── qwen25-7b/         ~4  GB  (Qwen2.5-7B-Instruct AWQ)
└── llama31-8b/        ~5  GB  (Meta-Llama-3.1-8B-Instruct AWQ)
```

Total Drive footprint: ~15 GB.

## How to run

Use `notebooks/02_quantize_models.ipynb` in Colab — it handles HF auth, GPU detection,
and Drive mounting. Or run from CLI in any Linux env with a GPU:

```bash
huggingface-cli login    # needed for gated LLaMA models
python -m scripts.quantize.quantize_awq \
    --root /content/drive/MyDrive/quest_kg/models \
    --models llama32-3b mistral-7b qwen25-7b llama31-8b
```

## HF gated models

LLaMA-3.1-8B-Instruct and LLaMA-3.2-3B-Instruct are **gated** on HuggingFace.
You must:
1. Visit each model page and click "Request access" (auto-approves for most users)
2. Run `huggingface-cli login` with a token that has read access

Mistral and Qwen are not gated.
