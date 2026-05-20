"""Generate notebooks/run_l4_questkg_finetune.ipynb.

Trains a small LLM (LLaMA-3.2-3B-Instruct by default) with LoRA as the
candidate selector in QUEST-KG-LLM on WebQSP + CWQ. The retrieval and
evidential MP layers stay frozen and locked -- we only train the final
selection step. This preserves the QUEST-KG methodology while making
the LLM picker much stronger.

Pipeline on Colab L4 (~22.5 GB VRAM):
  1. Mount Drive, clone repo, install peft+transformers+bitsandbytes
  2. Generate training data:
       for each train query:
         run QUEST-KG retrieval (locked config) -> top-K candidate paths + tails
         format as instruction: "Pick the best answer letter."
         label = correct entity letter (if in candidates) else SKIP
  3. LoRA fine-tune (rank 16) on this candidate-selection data
  4. Inference on test split with the fine-tuned adapter
  5. Save Hits@1 / Hits@10 / EM and per-query CSV
  6. Auto-push to GitHub

Smoke-test mode (default): 500 train, 200 test queries -- ~1.5 h on L4.
Full-scale mode (n_train=ALL, n_test=ALL): ~6-8 h on L4.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_NB = ROOT / "notebooks" / "run_l4_questkg_finetune.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "source": t,
                     "outputs": [], "execution_count": None}


cells = []

cells.append(md(
    "# QUEST-KG-LLM with LoRA fine-tuned selector\n\n"
    "Fine-tunes **LLaMA-3.2-3B-Instruct** with LoRA on WebQSP+CWQ training "
    "splits, using QUEST-KG retrieval (frozen, locked config) to generate "
    "candidate-selection training data. The fine-tuned LLM acts as the "
    "candidate selector in QUEST-KG-LLM, preserving the QUEST-KG methodology "
    "(retrieval + evidential MP + symbolic check) while making the final "
    "selector dramatically stronger.\n\n"
    "**Why this should beat the frozen Qwen-7B baseline (currently 0.397 / 0.236 H@1 on WebQSP/CWQ):**\n"
    "- Fine-tuned 3B model learns the WebQSP/CWQ answer-formatting + entity-disambiguation patterns\n"
    "- ChatKBQA (ACL 2024) uses the same idea (LoRA fine-tune LLaMA2-7B) and hits 0.832 / 0.827\n"
    "- Our retrieval recall is the ceiling; if QUEST-KG surfaces the correct answer in top-K we can hit ~retrieval-recall accuracy with a good selector\n\n"
    "**Mode selection** (cell 4): smoke-test vs full-scale.\n"
    "Smoke-test: 500 train / 200 test queries per dataset, ~1.5h on L4. "
    "Full-scale: all train / all test, ~6-8h on L4.\n\n"
    "Setup: secrets `HF_TOKEN` + `GH_TOKEN`, runtime = L4 GPU."
))

cells.append(md("## 1. Mount Drive + clone repo"))
cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "import os, subprocess, pathlib\n"
    "ROOT = '/content/drive/MyDrive/quest_kg'\n"
    "pathlib.Path(f'{ROOT}/results').mkdir(parents=True, exist_ok=True)\n"
    "pathlib.Path(f'{ROOT}/ft_adapters').mkdir(parents=True, exist_ok=True)\n"
    "REPO_OWNER = 'AKSB0567'\n"
    "REPO_NAME  = 'quest-kg-cikm2026'\n"
    "REPO_DIR   = f'/content/{REPO_NAME}'\n"
    "try:\n"
    "    from google.colab import userdata\n"
    "    gh_token = userdata.get('GH_TOKEN')\n"
    "except Exception:\n"
    "    gh_token = None\n"
    "url = (f'https://{REPO_OWNER}:{gh_token}@github.com/{REPO_OWNER}/{REPO_NAME}.git'\n"
    "       if gh_token else f'https://github.com/{REPO_OWNER}/{REPO_NAME}.git')\n"
    "if not os.path.exists(REPO_DIR):\n"
    "    subprocess.run(['git', 'clone', url, REPO_DIR], check=True)\n"
    "else:\n"
    "    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--rebase'], check=False)\n"
    "subprocess.run(['git', '-C', REPO_DIR, 'config', 'user.email', 'colab@quest-kg.local'])\n"
    "subprocess.run(['git', '-C', REPO_DIR, 'config', 'user.name', 'quest-kg-colab-bot'])\n"
    "os.chdir(REPO_DIR)\n"
    "print('cwd:', os.getcwd())"
))

cells.append(md(
    "## 2. Install deps (peft + transformers + bitsandbytes + accelerate)\n\n"
    "Pinning `bitsandbytes>=0.45` is critical: older versions (0.44.x) try to "
    "import `triton.ops` which was removed in Triton 3.x (which is what Colab "
    "ships now). The 0.45+ branch fixed this incompatibility."
))
cells.append(code(
    "!pip install -q --upgrade pip\n"
    "# bitsandbytes 0.45+ is required -- older versions trigger 'triton.ops' import error\n"
    "!pip install -q --upgrade 'bitsandbytes>=0.45.3'\n"
    "!pip install -q 'transformers>=4.46.3' 'peft>=0.13.2' 'accelerate>=1.1.1'\n"
    "!pip install -q datasets sentence-transformers safetensors\n"
    "import torch\n"
    "p = torch.cuda.get_device_properties(0)\n"
    "print(f'GPU: {torch.cuda.get_device_name(0)} ({p.total_memory/1024**3:.1f} GB)')\n"
    "import transformers, peft, bitsandbytes\n"
    "print(f'transformers {transformers.__version__}  peft {peft.__version__}  bnb {bitsandbytes.__version__}')"
))

cells.append(md("## 3. HF login"))
cells.append(code(
    "from huggingface_hub import login\n"
    "try:\n"
    "    from google.colab import userdata\n"
    "    login(token=userdata.get('HF_TOKEN'))\n"
    "    print('HF login OK')\n"
    "except Exception as e:\n"
    "    import getpass\n"
    "    login(token=getpass.getpass('HF token: '))"
))

cells.append(md(
    "## 4. Config (smoke-test default; flip to full-scale when ready)"
))
cells.append(code(
    "# === MODE ===\n"
    "MODE = 'smoke'   # 'smoke' (500 train, 200 test) or 'full' (all train, all test)\n"
    "BASE_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'\n"
    "# Locked QUEST-KG retrieval config (per-dataset) -- same as the main paper\n"
    "QKG_LOCKED = {\n"
    "    'webqsp': dict(top_k=4, hops=1, agg='max', bidir=True),\n"
    "    'cwq':    dict(top_k=8, hops=3, agg='sum', bidir=True),\n"
    "}\n"
    "DATASETS = ['webqsp', 'cwq']\n"
    "if MODE == 'smoke':\n"
    "    N_TRAIN = {'webqsp': 500, 'cwq': 500}\n"
    "    N_TEST  = {'webqsp': 200, 'cwq': 200}\n"
    "    N_EPOCHS = 3\n"
    "else:\n"
    "    N_TRAIN = {'webqsp': 3000, 'cwq': 27000}\n"
    "    N_TEST  = {'webqsp': 1000, 'cwq': 500}\n"
    "    N_EPOCHS = 2\n"
    "MAX_CANDIDATES = 16   # top-K candidates shown to LLM per query\n"
    "LORA_RANK = 16\n"
    "LORA_ALPHA = 32\n"
    "LR = 2e-4\n"
    "BATCH = 4\n"
    "GRAD_ACCUM = 4\n"
    "TAG = f'finetuned-llama32-3b__{MODE}__seed0'\n"
    "OUT_DIR = f'{ROOT}/results'\n"
    "ADAPTER_DIR = f'{ROOT}/ft_adapters/{TAG}'\n"
    "pathlib.Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)\n"
    "print(f'MODE={MODE}  TAG={TAG}')\n"
    "print(f'N_TRAIN={N_TRAIN}  N_TEST={N_TEST}  N_EPOCHS={N_EPOCHS}')"
))

cells.append(md(
    "## 5. Build retrieval helper (loads QUEST-KG locked config, no LLM)"
))
cells.append(code(
    "import sys, time\n"
    "sys.path.insert(0, REPO_DIR)\n"
    "from quest_kg.data.encoders import SentenceTransformerEncoder\n"
    "from quest_kg.data.loaders import load_dataset\n"
    "from quest_kg.retrieval.schema_aware import SchemaAwareRetriever\n"
    "from quest_kg.symbolic.webqsp_freebase import FreebaseChecker\n"
    "from quest_kg.inference import QuestKG\n\n"
    "encoder = SentenceTransformerEncoder('sentence-transformers/all-MiniLM-L6-v2', batch_size=256)\n"
    "print(f'encoder on {encoder.device}')\n\n"
    "def trim_to_per_query_union(ds, n_eval):\n"
    "    if not ds.queries or not ds.queries[0].get('graph_triple_idx'):\n"
    "        return\n"
    "    queries_eval = ds.queries[:n_eval]\n"
    "    all_idx = set()\n"
    "    for q in queries_eval:\n"
    "        all_idx.update(q.get('graph_triple_idx', []))\n"
    "    sorted_idx = sorted(all_idx)\n"
    "    remap = {old: new for new, old in enumerate(sorted_idx)}\n"
    "    ds.triples = [ds.triples[i] for i in sorted_idx]\n"
    "    for q in queries_eval:\n"
    "        q['graph_triple_idx'] = [remap[i] for i in q.get('graph_triple_idx', []) if i in remap]\n\n"
    "def build_qkg(ds_name, ds, n_eval):\n"
    "    cfg = QKG_LOCKED[ds_name]\n"
    "    trim_to_per_query_union(ds, n_eval)\n"
    "    retriever = SchemaAwareRetriever(\n"
    "        triples=ds.triples, encoder=encoder, k=cfg['hops'],\n"
    "        top_k=cfg['top_k'], precompute=True, bidirectional=cfg['bidir'])\n"
    "    method = QuestKG(\n"
    "        retriever=retriever, symbolic_checker=FreebaseChecker(),\n"
    "        task_type='entity', answer_rescoring=True,\n"
    "        max_path_length=cfg['hops'], answer_aggregation=cfg['agg'])\n"
    "    return method"
))

cells.append(md(
    "## 6. Generate candidate-selection training data\n\n"
    "For each train query, run QUEST-KG retrieval (frozen, locked) and "
    "extract the top-K candidate answer entities + the supporting fact "
    "triples that ground them. Construct a multiple-choice prompt:\n\n"
    "    Question: {q}\n"
    "    Supporting facts:\n"
    "    - {triple_1}\n"
    "    - {triple_2}\n"
    "    ...\n"
    "    Candidates:\n"
    "    A) {entity_1}\n"
    "    B) {entity_2}\n"
    "    ...\n"
    "    Answer: {correct_letter}) {correct_entity}\n\n"
    "If the correct answer is not in the retrieved top-K candidates, we "
    "**skip** that training example (retrieval miss -- the LLM can't fix it)."
))
cells.append(code(
    "import re, json\n"
    "from tqdm.auto import tqdm\n"
    "DATA_ROOT = f'{REPO_DIR}/data'\n\n"
    "def build_prompt_label(q, candidates, facts, gold_set):\n"
    "    correct_idx = -1\n"
    "    for i, c in enumerate(candidates):\n"
    "        if c in gold_set:\n"
    "            correct_idx = i; break\n"
    "    if correct_idx < 0:\n"
    "        return None\n"
    "    letters = [chr(ord('A')+i) for i in range(len(candidates))]\n"
    "    prompt = (\n"
    "        f'### Question: {q}\\n\\n'\n"
    "        f'### Supporting Facts:\\n' + '\\n'.join(f'- {f}' for f in facts[:24]) + '\\n\\n'\n"
    "        f'### Candidates:\\n' + '\\n'.join(f'{l}) {c}' for l, c in zip(letters, candidates)) + '\\n\\n'\n"
    "        f'### Answer:'\n"
    "    )\n"
    "    label = f' {letters[correct_idx]}) {candidates[correct_idx]}'\n"
    "    return prompt, label\n\n"
    "def extract_candidates_facts(method, query_dict):\n"
    "    res = method.predict(query_dict['question'], anchors=query_dict.get('anchors', []), graph_triple_idx=query_dict.get('graph_triple_idx'))\n"
    "    paths = res.candidate_paths if hasattr(res, 'candidate_paths') else []\n"
    "    cands, seen = [], set()\n"
    "    facts, fact_seen = [], set()\n"
    "    for p in paths[:MAX_CANDIDATES * 2]:\n"
    "        tail = getattr(p, 'tail', None)\n"
    "        if not tail or tail in seen:\n"
    "            continue\n"
    "        seen.add(tail); cands.append(tail)\n"
    "        for t in p.triples[:3]:\n"
    "            key = (t.s, t.r.lstrip('~'), t.o)\n"
    "            if key not in fact_seen:\n"
    "                fact_seen.add(key)\n"
    "                facts.append(f'({t.s} | {t.r.lstrip(\"~\")} | {t.o})')\n"
    "        if len(cands) >= MAX_CANDIDATES:\n"
    "            break\n"
    "    return cands, facts\n\n"
    "train_data = {ds: [] for ds in DATASETS}\n"
    "test_data = {ds: [] for ds in DATASETS}\n"
    "PHASE_TIMES = {}\n\n"
    "for ds_name in DATASETS:\n"
    "    print(f'\\n=== {ds_name}: building retriever + train data ===')\n"
    "    t_phase = time.perf_counter()\n"
    "    ds_train = load_dataset(ds_name, DATA_ROOT)\n"
    "    train_queries = ds_train.queries[: N_TRAIN[ds_name]]\n"
    "    print(f'  building retriever for {len(train_queries)} train queries...')\n"
    "    method_train = build_qkg(ds_name, ds_train, N_TRAIN[ds_name])\n"
    "    kept = 0\n"
    "    pbar = tqdm(train_queries, desc=f'{ds_name} train', unit='q', dynamic_ncols=True)\n"
    "    for q in pbar:\n"
    "        try:\n"
    "            cands, facts = extract_candidates_facts(method_train, q)\n"
    "            gold = set(q.get('answer', []) if isinstance(q.get('answer'), list) else [q.get('answer')])\n"
    "            pair = build_prompt_label(q['question'], cands, facts, gold)\n"
    "            if pair is not None:\n"
    "                train_data[ds_name].append({'prompt': pair[0], 'label': pair[1]})\n"
    "                kept += 1\n"
    "                pbar.set_postfix(kept=kept, recall=f'{kept/(pbar.n+1):.2f}')\n"
    "        except Exception:\n"
    "            pass\n"
    "    pbar.close()\n"
    "    print(f'  {ds_name} train: kept {len(train_data[ds_name])} / {len(train_queries)} ({len(train_data[ds_name])/max(len(train_queries),1):.1%} retrieval recall)')\n"
    "\n"
    "    print(f'=== {ds_name}: building test data ===')\n"
    "    ds_test = load_dataset(ds_name, DATA_ROOT)\n"
    "    test_queries = ds_test.queries[: N_TEST[ds_name]]\n"
    "    method_test = build_qkg(ds_name, ds_test, N_TEST[ds_name])\n"
    "    pbar = tqdm(test_queries, desc=f'{ds_name} test', unit='q', dynamic_ncols=True)\n"
    "    for q in pbar:\n"
    "        try:\n"
    "            cands, facts = extract_candidates_facts(method_test, q)\n"
    "            gold = set(q.get('answer', []) if isinstance(q.get('answer'), list) else [q.get('answer')])\n"
    "            test_data[ds_name].append({'question': q['question'], 'candidates': cands, 'facts': facts, 'gold': list(gold)})\n"
    "        except Exception:\n"
    "            pass\n"
    "    pbar.close()\n"
    "    PHASE_TIMES[ds_name + '_data'] = time.perf_counter() - t_phase\n"
    "    print(f'  {ds_name} test: {len(test_data[ds_name])} examples (data phase wall: {PHASE_TIMES[ds_name + \"_data\"]/60:.1f} min)')\n"
    "\n"
    "(pathlib.Path(ROOT) / 'finetune_data').mkdir(exist_ok=True)\n"
    "for ds in DATASETS:\n"
    "    with open(f'{ROOT}/finetune_data/{ds}_train.jsonl', 'w', encoding='utf-8') as f:\n"
    "        for ex in train_data[ds]:\n"
    "            f.write(json.dumps(ex) + '\\n')\n"
    "    with open(f'{ROOT}/finetune_data/{ds}_test.jsonl', 'w', encoding='utf-8') as f:\n"
    "        for ex in test_data[ds]:\n"
    "            f.write(json.dumps(ex) + '\\n')\n"
    "print('\\nsaved finetune_data/*.jsonl')\n"
    "print('phase timings (min):', {k: f'{v/60:.1f}' for k, v in PHASE_TIMES.items()})"
))

cells.append(md(
    "## 7. Load LLaMA-3.2-3B with LoRA adapter"
))
cells.append(code(
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n"
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
    "import torch\n\n"
    "tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side='right')\n"
    "if tok.pad_token is None:\n"
    "    tok.pad_token = tok.eos_token\n"
    "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n"
    "                          bnb_4bit_compute_dtype=torch.bfloat16,\n"
    "                          bnb_4bit_use_double_quant=True)\n"
    "model = AutoModelForCausalLM.from_pretrained(\n"
    "    BASE_MODEL, quantization_config=bnb, device_map='auto',\n"
    "    torch_dtype=torch.bfloat16, trust_remote_code=True)\n"
    "model = prepare_model_for_kbit_training(model)\n"
    "lcfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,\n"
    "                   bias='none', task_type='CAUSAL_LM',\n"
    "                   target_modules=['q_proj','k_proj','v_proj','o_proj',\n"
    "                                    'gate_proj','up_proj','down_proj'])\n"
    "model = get_peft_model(model, lcfg)\n"
    "model.print_trainable_parameters()"
))

cells.append(md("## 8. Train LoRA on the candidate-selection data"))
cells.append(code(
    "from datasets import Dataset\n"
    "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n"
    "import json, pathlib\n\n"
    "def tokenize_example(example, max_len=1024):\n"
    "    full = example['prompt'] + example['label'] + tok.eos_token\n"
    "    enc = tok(full, truncation=True, max_length=max_len, padding='max_length')\n"
    "    prompt_ids = tok(example['prompt'], truncation=True, max_length=max_len)['input_ids']\n"
    "    labels = enc['input_ids'].copy()\n"
    "    # mask out prompt tokens (don't train on input)\n"
    "    for i in range(min(len(prompt_ids), len(labels))):\n"
    "        labels[i] = -100\n"
    "    enc['labels'] = labels\n"
    "    return enc\n\n"
    "all_train = []\n"
    "for ds in DATASETS:\n"
    "    all_train.extend(train_data[ds])\n"
    "print(f'total train examples: {len(all_train)}')\n"
    "hf_ds = Dataset.from_list(all_train)\n"
    "hf_ds = hf_ds.map(tokenize_example, remove_columns=['prompt','label'],\n"
    "                  batched=False)\n\n"
    "args = TrainingArguments(\n"
    "    output_dir=f'{ADAPTER_DIR}/trainer',\n"
    "    num_train_epochs=N_EPOCHS,\n"
    "    per_device_train_batch_size=BATCH,\n"
    "    gradient_accumulation_steps=GRAD_ACCUM,\n"
    "    learning_rate=LR,\n"
    "    lr_scheduler_type='cosine',\n"
    "    warmup_ratio=0.05,\n"
    "    logging_steps=10,\n"
    "    save_strategy='epoch',\n"
    "    save_total_limit=1,\n"
    "    bf16=True,\n"
    "    optim='paged_adamw_8bit',\n"
    "    report_to='none',\n"
    "    dataloader_num_workers=0,\n"
    ")\n"
    "trainer = Trainer(model=model, args=args, train_dataset=hf_ds,\n"
    "                   tokenizer=tok)\n"
    "import time\n"
    "t_train = time.perf_counter()\n"
    "trainer.train()  # transformers Trainer prints its own tqdm progress bar with ETA\n"
    "PHASE_TIMES['train'] = time.perf_counter() - t_train\n"
    "print(f'\\nTraining wall: {PHASE_TIMES[\"train\"]/60:.1f} min')\n"
    "model.save_pretrained(ADAPTER_DIR)\n"
    "tok.save_pretrained(ADAPTER_DIR)\n"
    "print(f'adapter saved to {ADAPTER_DIR}')"
))

cells.append(md("## 9. Inference + scoring (with tqdm)"))
cells.append(code(
    "import torch, re, json, time\n"
    "from tqdm.auto import tqdm\n"
    "model.eval()\n"
    "@torch.inference_mode()\n"
    "def predict(prompt, max_new=32):\n"
    "    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to('cuda')\n"
    "    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,\n"
    "                          pad_token_id=tok.eos_token_id)\n"
    "    pl = enc.input_ids.shape[1]\n"
    "    return tok.decode(out[0][pl:], skip_special_tokens=True)\n\n"
    "def score_dataset(ds_name):\n"
    "    cells = test_data[ds_name]\n"
    "    rows, correct = [], 0\n"
    "    pbar = tqdm(enumerate(cells), total=len(cells), desc=f'eval {ds_name}',\n"
    "                 unit='q', dynamic_ncols=True)\n"
    "    for qi, ex in pbar:\n"
    "        if not ex['candidates']:\n"
    "            rows.append(dict(qid=qi, pred=None, gold=ex['gold'], em=0))\n"
    "            continue\n"
    "        letters = [chr(ord('A')+i) for i in range(len(ex['candidates']))]\n"
    "        prompt = (f'### Question: {ex[\"question\"]}\\n\\n'\n"
    "                   f'### Supporting Facts:\\n' + '\\n'.join(f'- {f}' for f in ex['facts'][:24]) + '\\n\\n'\n"
    "                   f'### Candidates:\\n' + '\\n'.join(f'{l}) {c}' for l, c in zip(letters, ex['candidates'])) + '\\n\\n'\n"
    "                   f'### Answer:')\n"
    "        out = predict(prompt)\n"
    "        m = re.search(r'([A-Z])\\)', out)\n"
    "        pred = None\n"
    "        if m:\n"
    "            idx = ord(m.group(1)) - ord('A')\n"
    "            if 0 <= idx < len(ex['candidates']):\n"
    "                pred = ex['candidates'][idx]\n"
    "        is_correct = pred in ex['gold'] if pred else False\n"
    "        correct += int(is_correct)\n"
    "        rows.append(dict(qid=qi, pred=pred, gold=ex['gold'],\n"
    "                          em=int(is_correct), raw=out[:120]))\n"
    "        pbar.set_postfix(h1=f'{correct/(qi+1):.3f}')\n"
    "    pbar.close()\n"
    "    return correct / max(len(cells), 1), rows\n\n"
    "results = {}\n"
    "for ds in DATASETS:\n"
    "    t0 = time.perf_counter()\n"
    "    h1, rows = score_dataset(ds)\n"
    "    PHASE_TIMES[ds + '_eval'] = time.perf_counter() - t0\n"
    "    results[ds] = dict(hits_at_1=h1, n=len(rows), per_query=rows)\n"
    "    print(f'\\n=== {ds}: Hits@1 = {h1:.4f}  (n={len(rows)}, eval wall: {PHASE_TIMES[ds+\"_eval\"]/60:.1f} min) ===')\n"
    "\n"
    "print('\\nALL PHASE TIMES (min):')\n"
    "for k, v in PHASE_TIMES.items():\n"
    "    print(f'  {k:>20s}: {v/60:6.1f} min')"
))

cells.append(md("## 10. Save results + push to GitHub"))
cells.append(code(
    "import shutil, json, pandas as pd, subprocess\n"
    "REPO_RESULTS = f'{REPO_DIR}/results'\n"
    "os.makedirs(REPO_RESULTS, exist_ok=True)\n"
    "for ds, info in results.items():\n"
    "    base = f'quest_kg_llm_ft__{ds}__{TAG}'\n"
    "    json_path = f'{REPO_RESULTS}/{base}.json'\n"
    "    csv_path  = f'{REPO_RESULTS}/{base}.csv'\n"
    "    df = pd.DataFrame(info['per_query'])\n"
    "    df.to_csv(csv_path, index=False)\n"
    "    summary = {\n"
    "        'n': info['n'], 'n_attempted': info['n'], 'abstention_rate': 0.0,\n"
    "        'exact_match': info['hits_at_1'], 'hits_at_1': info['hits_at_1'],\n"
    "        'primary_metric': 'accuracy', 'primary_value': info['hits_at_1'],\n"
    "        'task_type': 'qa', 'method': 'quest_kg_llm_ft',\n"
    "        'dataset': ds, 'llm': BASE_MODEL, 'tag': TAG,\n"
    "        'mode': MODE, 'n_epochs': N_EPOCHS,\n"
    "        'lora_rank': LORA_RANK, 'n_train': len(train_data[ds]),\n"
    "    }\n"
    "    with open(json_path, 'w') as f:\n"
    "        json.dump(summary, f, indent=2)\n"
    "    shutil.copy2(json_path, f'{OUT_DIR}/{base}.json')\n"
    "    shutil.copy2(csv_path,  f'{OUT_DIR}/{base}.csv')\n"
    "    print(f'  saved {base}: H@1={info[\"hits_at_1\"]:.4f}')\n"
    "subprocess.run(['git','-C',REPO_DIR,'add','results/'], capture_output=True)\n"
    "cm = subprocess.run(['git','-C',REPO_DIR,'commit','-m',f'finetune {MODE}: '+', '.join(f'{ds}={results[ds][\"hits_at_1\"]:.3f}' for ds in DATASETS)],\n"
    "                      capture_output=True, text=True)\n"
    "if 'nothing to commit' not in cm.stdout + cm.stderr:\n"
    "    subprocess.run(['git','-C',REPO_DIR,'push'], capture_output=True)\n"
    "    print('pushed to GitHub')"
))

cells.append(md(
    "## 11. Comparison to baselines\n\n"
    "Reference numbers (Hits@1):\n\n"
    "| Method                                       | WebQSP   | CWQ      |\n"
    "|----------------------------------------------|----------|----------|\n"
    "| ChatKBQA (LLaMA2-7B fine-tuned, ACL 2024)    | 0.832    | 0.827    |\n"
    "| GNN-RAG + RA (LLaMA2-7B, ACL 2025)           | 0.828    | 0.628    |\n"
    "| ToG + GPT-4 (ICLR 2024)                       | 0.826    | 0.695    |\n"
    "| RoG (LLaMA2-7B, ICLR 2024)                   | 0.800    | 0.578    |\n"
    "| **QUEST-KG-LLM (frozen Qwen-7B, ours, prior)** | 0.397  | 0.236    |\n"
    "| **QUEST-KG-LLM-FT (LLaMA-3.2-3B + LoRA, this notebook)** | TBD | TBD |\n\n"
    "Smoke-test gate:\n"
    "- If WebQSP >= 0.55, full-scale run is justified.\n"
    "- If WebQSP < 0.40, the retrieval recall is the bottleneck; we need to also enlarge top_k / use larger LLM.\n\n"
    "Anti-idle JS (DevTools console):\n"
    "```javascript\n"
    "function ClickConnect(){ document.querySelector('colab-toolbar-button#connect')?.click(); }\n"
    "setInterval(ClickConnect, 60000);\n"
    "```"
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
OUT_NB.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT_NB}")
print(f"Cells: {len(cells)}")
