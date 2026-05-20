"""Generate 2 Colab L4 notebooks for the EA push:
  notebooks/run_l4_ea_block1.ipynb  -> OpenEA IDS D-W-100K + D-Y-100K (monolingual EN)
  notebooks/run_l4_ea_block2.ipynb  -> OpenEA IDS EN-FR-100K + EN-DE-100K (cross-lingual)

Each notebook:
  1. Mount Drive, clone repo, install deps
  2. Download OpenEA v2.0 dataset (Dropbox direct link)
  3. For each of its 2 sub-datasets: run scripts/ea_runner.py
  4. Push per-cell results to GitHub
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def make_notebook(block_name: str, sub_datasets: list[tuple[str, str, str]]):
    """sub_datasets: list of (display_name, openea_folder_name, cross_lingual_flag)
                     e.g., ('ids-d-w', 'D_W_100K_V2', 'monolingual')
    """
    cells = []
    cells.append(md(
        f"# {block_name} — QUEST-KG entity alignment on OpenEA IDS100K\n\n"
        f"Runs QUEST-KG-EA on the following OpenEA IDS-100K-V2 sub-datasets:\n"
        + "".join(f"\n- **{d}** (folder `{f}`, {kind})" for d, f, kind in sub_datasets)
        + "\n\n"
        "**Setup**: HF_TOKEN + GH_TOKEN in Colab secrets, runtime = L4 GPU.\n"
        "**Total wall**: ~20–40 min per dataset (no LLM needed, MiniLM encoding + cosine).\n"
        "**Output**: per-cell JSON + CSV, auto-pushed to GitHub."
    ))

    cells.append(md("## 1. Mount Drive + clone repo"))
    cells.append(code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import os, subprocess, pathlib\n"
        "ROOT = '/content/drive/MyDrive/quest_kg'\n"
        "pathlib.Path(f'{ROOT}/data/raw/openea').mkdir(parents=True, exist_ok=True)\n"
        "pathlib.Path(f'{ROOT}/results').mkdir(parents=True, exist_ok=True)\n\n"
        "REPO_OWNER = 'AKSB0567'\n"
        "REPO_NAME  = 'quest-kg-cikm2026'\n"
        "REPO_DIR   = f'/content/{REPO_NAME}'\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    gh_token = userdata.get('GH_TOKEN')\n"
        "    print('GH_TOKEN found')\n"
        "except Exception:\n"
        "    gh_token = None\n"
        "    print('GH_TOKEN not set; results will NOT auto-push to GitHub (still saved to Drive)')\n"
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

    cells.append(md("## 2. Install deps + verify CUDA"))
    cells.append(code(
        "!pip install -q sentence-transformers torch pandas numpy\n"
        "import torch\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "print(f'GPU: {torch.cuda.get_device_name(0)} ({p.total_memory/1024**3:.1f} GB)')"
    ))

    cells.append(md(
        "## 3. Download OpenEA v2.0 dataset (~1 GB)\n\n"
        "Skipped if already extracted on Drive. Uses the Dropbox direct link from\n"
        "the OpenEA README. If Dropbox is unreachable, replace the URL with the\n"
        "figshare mirror: `https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760`."
    ))
    cells.append(code(
        "import os, subprocess, pathlib\n"
        "OEA_DIR = f'{ROOT}/data/raw/openea'\n"
        "ZIP_PATH = f'{OEA_DIR}/openea_v2.zip'\n"
        "MARKER = f'{OEA_DIR}/.extracted'\n"
        "if not os.path.exists(MARKER):\n"
        "    if not os.path.exists(ZIP_PATH):\n"
        "        print('Downloading OpenEA_dataset_v2.0.zip ...')\n"
        "        subprocess.run([\n"
        "            'curl', '-sL', '-o', ZIP_PATH, '--max-time', '900',\n"
        "            'https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v2.0.zip?dl=1'\n"
        "        ], check=True)\n"
        "    print('Unzipping ...')\n"
        "    subprocess.run(['unzip', '-q', '-o', ZIP_PATH, '-d', OEA_DIR], check=True)\n"
        "    pathlib.Path(MARKER).touch()\n"
        "    print('extracted ->', OEA_DIR)\n"
        "else:\n"
        "    print('already extracted')\n"
        "print(os.listdir(OEA_DIR)[:10])"
    ))

    cells.append(md("## 4. Locate the OpenEA dataset folder"))
    cells.append(code(
        "import os\n"
        "from pathlib import Path\n"
        "def find_openea_folder(root: str, folder_name: str) -> str:\n"
        "    '''OpenEA zip layout varies. Search for the requested sub-dataset folder.'''\n"
        "    root_p = Path(root)\n"
        "    for cand in root_p.rglob(folder_name):\n"
        "        if cand.is_dir() and any(cand.glob('rel_triples_1')):\n"
        "            return str(cand)\n"
        "    raise FileNotFoundError(f'Could not find OpenEA folder {folder_name} under {root}')\n"
        "OEA_BASE = f'{ROOT}/data/raw/openea'\n"
        "print('Sample contents:', list(Path(OEA_BASE).iterdir())[:5])"
    ))

    cells.append(md("## 5. Run QUEST-KG-EA on each sub-dataset"))
    runner_code = (
        "import subprocess, json, os, shutil\n"
        "RESULTS = f'{ROOT}/results'\n"
        "REPO_RESULTS = f'{REPO_DIR}/results'\n"
        "os.makedirs(RESULTS, exist_ok=True); os.makedirs(REPO_RESULTS, exist_ok=True)\n\n"
        "DATASETS = [\n"
    )
    for display, folder, kind in sub_datasets:
        runner_code += f"  ({display!r}, {folder!r}, {kind!r}),\n"
    runner_code += (
        "]\n"
        "TAG = 'colab-l4__symbolic__seed0'\n"
        "# Cross-lingual datasets need a multilingual encoder\n"
        "CROSS_LINGUAL = ('cross-lingual' in ' '.join(k for _, _, k in DATASETS))\n"
        "ENCODER = ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'\n"
        "           if CROSS_LINGUAL\n"
        "           else 'sentence-transformers/all-MiniLM-L6-v2')\n"
        "print(f'Encoder: {ENCODER}')\n"
        "for display, folder, kind in DATASETS:\n"
        "    print(f'\\n=== {display} ({kind}) ===')\n"
        "    ds_root = find_openea_folder(OEA_BASE, folder)\n"
        "    print(f'  root: {ds_root}')\n"
        "    out_json = f'{REPO_RESULTS}/quest_kg_ea__{display}__{TAG}.json'\n"
        "    if os.path.exists(out_json):\n"
        "        print(f'  SKIP (exists): {out_json}')\n"
        "        continue\n"
        "    r = subprocess.run([\n"
        "        'python', 'scripts/ea_runner.py',\n"
        "        '--dataset', display,\n"
        "        '--root', ds_root,\n"
        "        '--format', 'openea',\n"
        "        '--n_sample', '1000',\n"
        "        '--tag', TAG,\n"
        "        '--encoder', ENCODER,\n"
        "        '--out_dir', REPO_RESULTS,\n"
        "    ], capture_output=True, text=True)\n"
        "    print(r.stdout[-2000:])\n"
        "    if r.returncode != 0:\n"
        "        print('  STDERR:', r.stderr[-1500:])\n"
        "        continue\n"
        "    # mirror to Drive\n"
        "    base = f'quest_kg_ea__{display}__{TAG}'\n"
        "    for suf in ('.json', '.csv'):\n"
        "        src = f'{REPO_RESULTS}/{base}{suf}'\n"
        "        if os.path.exists(src):\n"
        "            shutil.copy2(src, f'{RESULTS}/{base}{suf}')\n"
        "    # commit + push\n"
        "    subprocess.run(['git', '-C', REPO_DIR, 'add', 'results/'], capture_output=True)\n"
        "    cm = subprocess.run(['git', '-C', REPO_DIR, 'commit', '-m', f'colab EA: {display}'],\n"
        "                          capture_output=True, text=True)\n"
        "    if 'nothing to commit' not in cm.stdout + cm.stderr:\n"
        "        push = subprocess.run(['git', '-C', REPO_DIR, 'push'], capture_output=True, text=True)\n"
        "        print(f'  pushed: {push.returncode == 0}')\n"
        "print('\\n=== all sub-datasets done ===')"
    )
    cells.append(code(runner_code))

    cells.append(md(
        "## Anti-idle JS (paste in DevTools console)\n\n"
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
    return nb


nb1 = make_notebook(
    "run_l4_ea_block1: OpenEA IDS100K monolingual (D-W, D-Y)",
    [
        ("ids-d-w", "D_W_100K_V2", "monolingual EN (DBpedia <-> Wikidata)"),
        ("ids-d-y", "D_Y_100K_V2", "monolingual EN (DBpedia <-> YAGO)"),
    ],
)
out1 = ROOT / "notebooks" / "run_l4_ea_block1.ipynb"
out1.write_text(json.dumps(nb1, indent=1))

nb2 = make_notebook(
    "run_l4_ea_block2: OpenEA IDS100K cross-lingual (EN-FR, EN-DE)",
    [
        ("ids-en-fr", "EN_FR_100K_V2", "cross-lingual (English <-> French)"),
        ("ids-en-de", "EN_DE_100K_V2", "cross-lingual (English <-> German)"),
    ],
)
out2 = ROOT / "notebooks" / "run_l4_ea_block2.ipynb"
out2.write_text(json.dumps(nb2, indent=1))

print(f"Wrote {out1}")
print(f"Wrote {out2}")
