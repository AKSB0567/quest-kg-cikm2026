"""Generates `notebooks/run_l4_gap_fill.ipynb` — the lean, resumable Colab L4
notebook that runs ONLY the experiments still missing for the CIKM 2026 submission.

What it runs:
  - WebQSP @ N=1000:  cok (pending), quest_kg_llm with LOCKED config
  - CWQ @ N=500:      vanilla_rag, graphrag, tog1, tog2, cok, quest_kg_llm (LOCKED)

What it skips:
  - OrgAccess (all 7 methods already in `results/`)
  - ICEWS18 (all 7 methods already in `results/`)
  - quest_kg (pure symbolic) on WebQSP/CWQ — already done locally with locked config

Safety:
  - Per-cell save to Drive AND git push immediately after each method finishes
  - Skip if JSON exists (idempotent across sessions)
  - Anti-idle JS snippet at the top
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "run_l4_gap_fill.ipynb"


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


cells = []

cells.append(md(
    "# run_l4_gap_fill — finish the missing Colab benchmarks\n\n"
    "Runs **only** the experiments still missing for the QUEST-KG CIKM 2026 paper.\n\n"
    "**Already done** (in `results/` already, this notebook skips them):\n"
    "- OrgAccess: quest_kg, quest_kg_llm, vanilla_rag, graphrag, tog1, tog2, cok (N=750)\n"
    "- ICEWS18: quest_kg, quest_kg_llm, vanilla_rag, graphrag, tog1, tog2, cok (N=200)\n"
    "- WebQSP: vanilla_rag, graphrag, tog1, tog2 (N=1000)\n"
    "- Local symbolic quest_kg on all 4 datasets (locked config, N matched)\n\n"
    "**This notebook runs**:\n"
    "- WebQSP: **cok** (didn't finish last session) + **quest_kg_llm** with locked config (previous run used stale pre-Iter-5b config and scored only 0.053)\n"
    "- CWQ: **quest_kg_llm** (locked) + **vanilla_rag, graphrag, tog1, tog2, cok** (all 5 LLM baselines at N=500)\n\n"
    "**Estimated wall (L4)**:\n"
    "- WebQSP cok ≈ 2.5h\n"
    "- WebQSP quest_kg_llm (locked) ≈ 10 min\n"
    "- CWQ baselines (5) ≈ 4-5h\n"
    "- CWQ quest_kg_llm ≈ 5 min\n"
    "- **Total ≈ 7-8h** (fits 12h session with headroom)\n\n"
    "**Per-cell save + git push** so a runtime disconnect never loses completed work.\n\n"
    "---\n"
    "**Before running:**\n"
    "1. Runtime → Change runtime type → **L4 GPU** (22.5 GB)\n"
    "2. Add `HF_TOKEN` to secrets (key icon, notebook access ON)\n"
    "3. Add `GH_TOKEN` to secrets (push-enabled PAT)\n"
    "4. Paste the anti-idle JS at the bottom into DevTools console\n"
    "5. Runtime → Run all"
))

cells.append(md("## 1. Mount Drive"))
cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n\n"
    "import os, pathlib\n"
    "ROOT = '/content/drive/MyDrive/quest_kg'\n"
    "for sub in ['data/raw', 'data/processed', 'models', 'results', 'logs']:\n"
    "    pathlib.Path(f'{ROOT}/{sub}').mkdir(parents=True, exist_ok=True)\n"
    "print('Drive root:', ROOT)"
))

cells.append(md("## 2. Clone / pull repo (always pull latest)"))
cells.append(code(
    "import os, subprocess\n"
    "REPO_OWNER = 'AKSB0567'\n"
    "REPO_NAME  = 'quest-kg-cikm2026'\n"
    "REPO_DIR   = f'/content/{REPO_NAME}'\n\n"
    "gh_token = None\n"
    "try:\n"
    "    from google.colab import userdata\n"
    "    gh_token = userdata.get('GH_TOKEN')\n"
    "    print('GH_TOKEN found (will be used for push)')\n"
    "except Exception:\n"
    "    print('GH_TOKEN not set; results will NOT be auto-pushed to GitHub')\n\n"
    "url = (f'https://{REPO_OWNER}:{gh_token}@github.com/{REPO_OWNER}/{REPO_NAME}.git'\n"
    "       if gh_token else f'https://github.com/{REPO_OWNER}/{REPO_NAME}.git')\n\n"
    "if not os.path.exists(REPO_DIR):\n"
    "    r = subprocess.run(['git', 'clone', url, REPO_DIR], capture_output=True, text=True)\n"
    "else:\n"
    "    r = subprocess.run(['git', '-C', REPO_DIR, 'pull', '--rebase'], capture_output=True, text=True)\n"
    "print(r.stdout); print(r.stderr)\n"
    "assert r.returncode == 0, f'git failed: {r.stderr}'\n"
    "# Configure git for auto-commits from this notebook\n"
    "subprocess.run(['git', '-C', REPO_DIR, 'config', 'user.email', 'colab@quest-kg.local'])\n"
    "subprocess.run(['git', '-C', REPO_DIR, 'config', 'user.name', 'quest-kg-colab-bot'])\n"
    "os.chdir(REPO_DIR)\n"
    "print('cwd:', os.getcwd())"
))

cells.append(md("## 3. Install dependencies"))
cells.append(code(
    "!pip install -q --upgrade pip\n"
    "!pip install -q requests tqdm gdown datasets\n"
    "!pip install -q transformers accelerate huggingface_hub sentence-transformers safetensors einops sentencepiece\n"
    "!pip install -q bitsandbytes\n"
    "!pip install -q pandas matplotlib seaborn"
))

cells.append(md("## 4. CUDA sanity"))
cells.append(code(
    "import torch\n"
    "p = torch.cuda.get_device_properties(0)\n"
    "print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
    "print(f'CC: sm_{p.major}{p.minor}  VRAM: {p.total_memory/1024**3:.1f} GB')\n"
    "print(f'torch: {torch.__version__}  cuda: {torch.version.cuda}')\n"
    "assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > L4'"
))

cells.append(md("## 5. HuggingFace login"))
cells.append(code(
    "from huggingface_hub import login, whoami\n"
    "try:\n"
    "    from google.colab import userdata\n"
    "    login(token=userdata.get('HF_TOKEN'))\n"
    "    print('HF login via secrets OK')\n"
    "except Exception as e:\n"
    "    print(f'secret manager unavailable ({e}); falling back to prompt')\n"
    "    import getpass\n"
    "    login(token=getpass.getpass('HF token: '))\n"
    "print('logged in as:', whoami().get('name'))"
))

cells.append(md("## 6. Data symlink (Drive → repo)"))
cells.append(code(
    "import os\n"
    "DATA_ROOT = f'{ROOT}/data'\n"
    "if not os.path.exists(f'{REPO_DIR}/data'):\n"
    "    os.symlink(DATA_ROOT, f'{REPO_DIR}/data')\n"
    "elif not os.path.islink(f'{REPO_DIR}/data') and not os.listdir(f'{REPO_DIR}/data'):\n"
    "    os.rmdir(f'{REPO_DIR}/data'); os.symlink(DATA_ROOT, f'{REPO_DIR}/data')\n\n"
    "for ds in ['orgaccess', 'icews18', 'webqsp', 'cwq']:\n"
    "    p = f'{DATA_ROOT}/raw/{ds}'\n"
    "    ok = os.path.exists(f'{p}/.done') or os.path.exists(f'{p}/dataset')\n"
    "    print(f'  {\"OK \" if ok else \"MISS\"} {ds}')\n\n"
    "missing = [ds for ds in ['webqsp','cwq']\n"
    "           if not (os.path.exists(f'{DATA_ROOT}/raw/{ds}/.done') or os.path.exists(f'{DATA_ROOT}/raw/{ds}/dataset'))]\n"
    "if missing:\n"
    "    print('Downloading missing:', missing)\n"
    "    !python -m scripts.download.download_all --root $DATA_ROOT\n"
    "else:\n"
    "    print('WebQSP + CWQ datasets present.')"
))

cells.append(md(
    "## 7. Config — ONLY the missing experiments\n\n"
    "Edit ONLY here to scope down further if you hit time pressure.\n"
    "Each method×dataset cell saves a JSON to Drive AND auto-commits to GitHub on success."
))
cells.append(code(
    "# === Per-dataset N (matched to what we have locally / from prior runs) ===\n"
    "LIMIT_QUERIES = {\n"
    "    'webqsp': 1000,   # matches the WebQSP baselines already run last session\n"
    "    'cwq':     500,   # matches our local quest_kg at N=500 + saves time\n"
    "}\n\n"
    "# === Baseline KG cap (matches the existing WebQSP runs) ===\n"
    "KG_SUBSET_BASELINE = {'webqsp': 100000, 'cwq': 100000}\n\n"
    "# === quest_kg_llm LOCKED config (from run_local_questkg_only.py) ===\n"
    "# WebQSP: top_k=4, hops=1, agg=max, bidir=True, per-query graphs (kg_subset=None)\n"
    "# CWQ:    top_k=8, hops=3, agg=sum, bidir=True, per-query graphs (kg_subset=None)\n"
    "QKG_LOCKED = {\n"
    "    'webqsp': dict(top_k=4, hops=1, agg='max', bidir=True),\n"
    "    'cwq':    dict(top_k=8, hops=3, agg='sum', bidir=True),\n"
    "}\n\n"
    "# === ONLY the methods we still need on each dataset ===\n"
    "TODO = {\n"
    "    'webqsp': ['quest_kg_llm', 'cok'],  # qkg_llm rerun with LOCKED config; cok pending\n"
    "    'cwq':    ['quest_kg_llm', 'vanilla_rag', 'graphrag', 'tog1', 'tog2', 'cok'],\n"
    "}\n\n"
    "# === FORCE RERUN: even if JSON exists in Drive from last session, RUN AGAIN ===\n"
    "# Session 1 ran quest_kg_llm on WebQSP with STALE pre-Iter-5b config (got 0.053).\n"
    "# We MUST overwrite that with the LOCKED config rerun. Same on CWQ if it exists.\n"
    "FORCE_RERUN = {\n"
    "    'webqsp': {'quest_kg_llm'},\n"
    "    'cwq':    {'quest_kg_llm'},\n"
    "}\n\n"
    "# === LLM (must match prior runs for fair comparison) ===\n"
    "LLM_ID = 'Qwen/Qwen2.5-7B-Instruct'\n"
    "ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2'\n"
    "SEED = 0\n"
    "TAG = f'colab-l4__{LLM_ID.split(\"/\")[-1]}__seed{SEED}'\n"
    "OUT_DIR = f'{ROOT}/results'\n"
    "REPO_OUT_DIR = f'{REPO_DIR}/results'\n"
    "os.makedirs(OUT_DIR, exist_ok=True)\n"
    "print(f'TAG = {TAG}')\n"
    "print(f'OUT = {OUT_DIR}  (mirrored to {REPO_OUT_DIR})')\n"
    "print(f'TODO (only what is missing):')\n"
    "for ds, ms in TODO.items():\n"
    "    print(f'  {ds}: {ms}')"
))

cells.append(md("## 8. Load encoder + LLM (once)"))
cells.append(code(
    "import sys, time, json, gc\n"
    "sys.path.insert(0, REPO_DIR)\n"
    "import warnings, logging\n"
    "warnings.filterwarnings('ignore')\n"
    "logging.getLogger('transformers').setLevel(logging.ERROR)\n"
    "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'\n\n"
    "from quest_kg.data.encoders import SentenceTransformerEncoder\n"
    "t0 = time.perf_counter()\n"
    "encoder = SentenceTransformerEncoder(ENCODER_ID)\n"
    "print(f'encoder ready in {time.perf_counter()-t0:.1f}s on {encoder.device}')\n\n"
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
    "t0 = time.perf_counter()\n"
    "tok = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True, padding_side='left')\n"
    "if tok.pad_token_id is None: tok.pad_token = tok.eos_token\n"
    "llm_model = AutoModelForCausalLM.from_pretrained(\n"
    "    LLM_ID, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True,\n"
    ")\n"
    "llm_model.eval()\n"
    "print(f'LLM {LLM_ID} ready in {time.perf_counter()-t0:.1f}s')\n"
    "print(f'VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB')\n\n"
    "class TinyLLM:\n"
    "    def __init__(self, m, t):\n"
    "        self.model, self.tok = m, t\n"
    "    @torch.inference_mode()\n"
    "    def generate(self, prompts, batch_size=2, max_new_tokens=24):\n"
    "        out = []\n"
    "        for i in range(0, len(prompts), batch_size):\n"
    "            b = prompts[i:i+batch_size]\n"
    "            enc = self.tok(b, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')\n"
    "            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)\n"
    "            pl = enc.input_ids.shape[1]\n"
    "            for j, seq in enumerate(gen):\n"
    "                out.append(self.tok.decode(seq[pl:], skip_special_tokens=True).strip())\n"
    "        return out\n\n"
    "llm = TinyLLM(llm_model, tok)\n"
    "print('gen smoke:', llm.generate(['The capital of France is'], batch_size=1, max_new_tokens=8)[0][:60])"
))

cells.append(md(
    "## 9. Helper: per-cell save + auto-push to GitHub\n\n"
    "Called after every (method, dataset) finishes. Saves JSON+CSV to Drive,\n"
    "copies into repo working tree, then commits and pushes. Idempotent — if push fails\n"
    "(token missing or rate limit), we keep going; data is safe in Drive either way."
))
cells.append(code(
    "import shutil, subprocess, json\n\n"
    "def save_and_push(json_path: str, csv_path: str, cell_tag: str):\n"
    "    # Mirror to repo working tree\n"
    "    os.makedirs(REPO_OUT_DIR, exist_ok=True)\n"
    "    for src in (json_path, csv_path):\n"
    "        if os.path.exists(src):\n"
    "            shutil.copy2(src, REPO_OUT_DIR)\n"
    "    # Stage + commit + push\n"
    "    try:\n"
    "        subprocess.run(['git', '-C', REPO_DIR, 'add', 'results/'], check=False, capture_output=True)\n"
    "        msg = f'colab gap-fill: {cell_tag}'\n"
    "        r = subprocess.run(['git', '-C', REPO_DIR, 'commit', '-m', msg],\n"
    "                           capture_output=True, text=True)\n"
    "        if r.returncode == 0:\n"
    "            push = subprocess.run(['git', '-C', REPO_DIR, 'push'],\n"
    "                                  capture_output=True, text=True)\n"
    "            if push.returncode == 0:\n"
    "                print(f'  [save_and_push] pushed: {cell_tag}')\n"
    "            else:\n"
    "                print(f'  [save_and_push] commit OK but push FAILED: {push.stderr[:200]}')\n"
    "        else:\n"
    "            # Nothing to commit (idempotent rerun) is normal\n"
    "            if 'nothing to commit' in (r.stdout + r.stderr).lower():\n"
    "                print(f'  [save_and_push] no changes for {cell_tag}')\n"
    "            else:\n"
    "                print(f'  [save_and_push] commit FAILED: {r.stderr[:200]}')\n"
    "    except Exception as e:\n"
    "        print(f'  [save_and_push] exception: {e}')\n\n"
    "print('save_and_push ready')"
))

cells.append(md(
    "## 10. Helper: build_method (with locked QUEST-KG-LLM config support)"
))
cells.append(code(
    "from quest_kg.retrieval.schema_aware import SchemaAwareRetriever\n"
    "from quest_kg.inference import QuestKG\n"
    "from quest_kg.symbolic.webqsp_freebase import FreebaseChecker\n"
    "from baselines.vanilla_rag import VanillaRAG\n"
    "from baselines.graphrag import GraphRAG\n"
    "from baselines.tog import ToG1, ToG2\n"
    "from baselines.chain_of_knowledge import ChainOfKnowledge\n\n"
    "QKG_TASK_TYPE = {'webqsp': 'entity', 'cwq': 'entity'}\n"
    "TASK_TYPE_MAP = {'webqsp': 'qa', 'cwq': 'qa'}\n\n"
    "def trim_to_per_query_union(ds, n_eval):\n"
    "    '''If queries have graph_triple_idx (rmanluo per-query graphs),\n"
    "    restrict ds.triples to the union of first n_eval queries graphs.'''\n"
    "    queries_eval = ds.queries[:n_eval]\n"
    "    if not queries_eval or not queries_eval[0].get('graph_triple_idx'):\n"
    "        return\n"
    "    all_idx = set()\n"
    "    for q in queries_eval:\n"
    "        all_idx.update(q.get('graph_triple_idx', []))\n"
    "    sorted_idx = sorted(all_idx)\n"
    "    remap = {old: new for new, old in enumerate(sorted_idx)}\n"
    "    ds.triples = [ds.triples[i] for i in sorted_idx]\n"
    "    for q in queries_eval:\n"
    "        q['graph_triple_idx'] = [remap[i] for i in q.get('graph_triple_idx', []) if i in remap]\n"
    "    print(f'  per-query KG union: {len(ds.triples)} triples')\n\n"
    "def build_for_quest_kg_llm(ds_name, ds, n_eval):\n"
    "    '''Builds retriever + checker + method for QUEST-KG-LLM using LOCKED config.'''\n"
    "    cfg = QKG_LOCKED[ds_name]\n"
    "    trim_to_per_query_union(ds, n_eval)\n"
    "    retriever = SchemaAwareRetriever(\n"
    "        triples=ds.triples, encoder=encoder, k=cfg['hops'],\n"
    "        top_k=cfg['top_k'], precompute=True, bidirectional=cfg['bidir'],\n"
    "    )\n"
    "    checker = FreebaseChecker()\n"
    "    method = QuestKG(\n"
    "        retriever=retriever, symbolic_checker=checker,\n"
    "        task_type=QKG_TASK_TYPE[ds_name],\n"
    "        answer_rescoring=True,\n"
    "        max_path_length=cfg['hops'],\n"
    "        answer_aggregation=cfg['agg'],\n"
    "        llm=llm,\n"
    "    )\n"
    "    return method, retriever\n\n"
    "def build_for_baseline(method_name, ds_name, ds):\n"
    "    '''Build a baseline. Uses the FULL kg_subset (100k) for fair comparison\n"
    "    with the WebQSP baselines already run last session.'''\n"
    "    kg_cap = KG_SUBSET_BASELINE.get(ds_name, 100000)\n"
    "    if kg_cap and len(ds.triples) > kg_cap:\n"
    "        ds.triples = ds.triples[:kg_cap]\n"
    "    if method_name == 'vanilla_rag': return VanillaRAG(ds.triples, encoder, llm, top_k=6)\n"
    "    if method_name == 'graphrag':    return GraphRAG(ds.triples, encoder, llm, k=2, top_k=16)\n"
    "    if method_name == 'tog1':        return ToG1(ds.triples, encoder, llm, max_depth=2, beam_width=3)\n"
    "    if method_name == 'tog2':        return ToG2(ds.triples, encoder, llm, max_depth=2, beam_width=3)\n"
    "    if method_name == 'cok':         return ChainOfKnowledge(ds.triples, encoder, llm, max_steps=2, top_k=4)\n"
    "    raise KeyError(method_name)\n\n"
    "print('build helpers ready')"
))

cells.append(md(
    "## 11. Main loop — run only the gap-fill cells, save+push after each\n\n"
    "Already-done methods skip immediately (JSON exists check)."
))
cells.append(code(
    "from quest_kg.data.loaders import load_dataset\n"
    "from quest_kg.eval.harness import aggregate, run_method, to_dataframe\n\n"
    "t_global = time.perf_counter()\n"
    "summary_rows = []\n\n"
    "for ds_name, methods in TODO.items():\n"
    "    print(f'\\n=== Dataset: {ds_name} ===')\n"
    "    n_limit = LIMIT_QUERIES[ds_name]\n"
    "    for method_name in methods:\n"
    "        tag = f'{method_name}__{ds_name}__{TAG}'\n"
    "        json_path = f'{OUT_DIR}/{tag}.json'\n"
    "        csv_path  = f'{OUT_DIR}/{tag}.csv'\n"
    "        force = method_name in FORCE_RERUN.get(ds_name, set())\n"
    "        if os.path.exists(json_path) and not force:\n"
    "            print(f'  SKIP {method_name} (already done)')\n"
    "            with open(json_path) as f:\n"
    "                summary_rows.append(json.load(f))\n"
    "            continue\n"
    "        if force and os.path.exists(json_path):\n"
    "            print(f'  FORCE RERUN {method_name} (overwriting stale {json_path})')\n"
    "            try:\n"
    "                os.rename(json_path, json_path + '.stale')\n"
    "                if os.path.exists(csv_path):\n"
    "                    os.rename(csv_path, csv_path + '.stale')\n"
    "            except Exception as e:\n"
    "                print(f'    rename warning: {e}')\n"
    "        print(f'  RUN  {method_name} ...', flush=True)\n"
    "        try:\n"
    "            # Fresh dataset load per cell — keeps ds.triples mutation isolated\n"
    "            ds = load_dataset(ds_name, str(f'{REPO_DIR}/data'))\n"
    "            print(f'    raw triples: {len(ds.triples)}, queries: {len(ds.queries)}')\n"
    "            queries = ds.queries[:n_limit]\n"
    "            t1 = time.perf_counter()\n"
    "            if method_name == 'quest_kg_llm':\n"
    "                method, retriever = build_for_quest_kg_llm(ds_name, ds, n_limit)\n"
    "                is_qkg = True\n"
    "                kg_cap = None  # per-query graphs\n"
    "                top_k_used = QKG_LOCKED[ds_name]['top_k']\n"
    "            else:\n"
    "                method = build_for_baseline(method_name, ds_name, ds)\n"
    "                is_qkg = False\n"
    "                kg_cap = KG_SUBSET_BASELINE[ds_name]\n"
    "                top_k_used = None\n"
    "            results = run_method(method, queries, method_name=method_name, is_questkg=is_qkg)\n"
    "            elapsed = time.perf_counter() - t1\n"
    "            agg = aggregate(results, task_type=TASK_TYPE_MAP[ds_name])\n"
    "            agg.update({\n"
    "                'method': method_name, 'dataset': ds_name,\n"
    "                'llm': LLM_ID, 'encoder': ENCODER_ID, 'seed': SEED,\n"
    "                'wallclock_s': round(elapsed, 1),\n"
    "                'limit': n_limit, 'kg_subset': kg_cap,\n"
    "                'top_k': top_k_used,\n"
    "                'tag': TAG,\n"
    "            })\n"
    "            if method_name == 'quest_kg_llm':\n"
    "                agg['quest_kg_llm_locked'] = True\n"
    "                agg['hops'] = QKG_LOCKED[ds_name]['hops']\n"
    "                agg['agg'] = QKG_LOCKED[ds_name]['agg']\n"
    "                agg['bidirectional'] = QKG_LOCKED[ds_name]['bidir']\n"
    "            df = to_dataframe(results)\n"
    "            df.to_csv(csv_path, index=False)\n"
    "            with open(json_path, 'w') as f:\n"
    "                json.dump(agg, f, indent=2)\n"
    "            pm = agg.get('primary_metric','?'); pv = agg.get('primary_value',0)\n"
    "            print(f'    OK {elapsed:.0f}s  EM={agg[\"exact_match\"]:.3f}  {pm}={pv:.3f}')\n"
    "            summary_rows.append(agg)\n"
    "            # Critical: save + push IMMEDIATELY so disconnect can't lose this result\n"
    "            save_and_push(json_path, csv_path, tag)\n"
    "        except Exception as e:\n"
    "            import traceback; traceback.print_exc()\n"
    "            print(f'    FAIL: {e}')\n"
    "        finally:\n"
    "            gc.collect(); torch.cuda.empty_cache()\n\n"
    "print(f'\\n=== Gap-fill done in {(time.perf_counter()-t_global)/60:.1f} min ===')"
))

cells.append(md("## 12. Summary table of everything (new + already-done)"))
cells.append(code(
    "import pandas as pd\n"
    "rows = []\n"
    "for agg in summary_rows:\n"
    "    rows.append({\n"
    "        'method': agg.get('method'),\n"
    "        'dataset': agg.get('dataset'),\n"
    "        'primary_metric': agg.get('primary_metric'),\n"
    "        'primary_value': agg.get('primary_value'),\n"
    "        'EM': agg.get('exact_match'),\n"
    "        'balanced_accuracy': agg.get('balanced_accuracy'),\n"
    "        'MRR': agg.get('mrr'),\n"
    "        'n': agg.get('n'),\n"
    "        'wallclock_s': agg.get('wallclock_s'),\n"
    "    })\n"
    "df_sum = pd.DataFrame(rows)\n"
    "pd.set_option('display.width', 200)\n"
    "pd.set_option('display.float_format', lambda x: f'{x:.3f}')\n"
    "print('\\n=== GAP-FILL SUMMARY ===')\n"
    "print(df_sum.to_string(index=False))\n\n"
    "if len(df_sum) > 0:\n"
    "    wide = df_sum.pivot_table(index='method', columns='dataset', values='primary_value', aggfunc='first')\n"
    "    print('\\n=== PRIMARY METRIC GRID ===')\n"
    "    print(wide.round(3))\n\n"
    "df_sum.to_csv(f'{OUT_DIR}/_l4_gap_fill_summary.csv', index=False)\n"
    "print(f'\\nSaved to {OUT_DIR}/_l4_gap_fill_summary.csv')\n\n"
    "# Final push\n"
    "save_and_push(f'{OUT_DIR}/_l4_gap_fill_summary.csv', '', '_l4_gap_fill_summary')"
))

cells.append(md(
    "## 13. Anti-idle JS (paste into DevTools console once)\n\n"
    "```javascript\n"
    "function ClickConnect(){\n"
    "  document.querySelector('colab-toolbar-button#connect')?.click();\n"
    "}\n"
    "setInterval(ClickConnect, 60000);\n"
    "```\n\n"
    "Open Chrome DevTools (F12) → Console → paste above → Enter. Keeps the\n"
    "runtime alive even if you tab away."
))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH}")
print(f"Cells: {len(cells)}")
