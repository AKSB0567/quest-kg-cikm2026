# Colab L4 — how to run the headline experiments

This is the **only** notebook you need to run on Colab. It does Phase 3 (matched-
condition headline), Phase 4 (hop + component ablations), and Phase 5 (calibration).

## One-time setup before first run

1. **Runtime → Change runtime type → L4 GPU** (22.5 GB VRAM required).
2. **Secrets** (left sidebar key icon, "Notebook access" ON):
   - `HF_TOKEN` — your HuggingFace token (read access to gated models).
   - `GH_TOKEN` — optional, only if the repo is private.
3. **Drive layout** — the notebook expects this on first run (already created by `setup_all.ipynb`):
   ```
   /content/drive/MyDrive/quest_kg/
       data/raw/{orgaccess,icews18,webqsp,cwq}/  # downloaded by setup_all
       results/                                   # outputs land here
       models/                                    # cached LLM/encoder
   ```

## Running the notebook

1. Open `notebooks/run_l4_everything.ipynb` in Colab.
2. **Runtime → Run all.**
3. Wait ~6-10 hours (idle in tab; paste the anti-idle JS from the bottom of
   the notebook into DevTools console to prevent disconnect).
4. When done: **File → Download → Download .ipynb** (the file with all cell
   outputs embedded).
5. Drop the downloaded `.ipynb` into the repo at `notebooks/run_l4_everything_OUTPUT.ipynb`
   (or just send back to dev). The cell outputs preserve every benchmark
   table, calibration row, ablation pivot — readable directly from the file.

## What the notebook produces

In `/content/drive/MyDrive/quest_kg/results/`:
- `<method>__<dataset>__colab-l4__Qwen2.5-7B-Instruct__seed0.{json,csv}` — per-cell
- `_l4_summary.csv` — consolidated headline grid
- `_l4_calibration.csv` — ECE + AURC per cell
- `_l4_hop_sweep.csv` — k ∈ {1,2,3,4} results for QUEST-KG
- `_l4_ablation_components.csv` — full / -bidir / -rel_bias / -ans_rescore

In the notebook's last cell (visible after download):
- Phase 3 headline pivot (method × dataset)
- Latency pivot
- Calibration table
- Hop sweep pivot
- Component-removal pivot

## If anything fails

The notebook is resumable: re-running skips any (method, dataset) cell whose
JSON already exists. If a single baseline crashes, the others continue.

Typical failures and fixes:
- **`HF_TOKEN` not found** → add it to Colab secrets and re-run cell 5.
- **OOM on 7B fp16 load** → downgrade `LLM_ID` in cell 14 (config) to
  `Qwen/Qwen2.5-3B-Instruct`. The full notebook will rerun automatically.
- **`git clone` exit 128** → confirm `GH_TOKEN` set, or make repo public.
- **L4 disconnect** → paste the anti-idle JS, then re-run "Run all" — it
  resumes from the first incomplete cell.
