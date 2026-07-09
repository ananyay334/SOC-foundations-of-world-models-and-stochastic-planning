"""Generate GRASP_Benchmark_Colab.ipynb (embeds the w6 scripts as writefile cells)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG, EPOCHS, BATCH, RUN = 112, 20, 96, "pusht/lewm"
CKPT = f"{RUN}/weights_epoch_{EPOCHS}.pt"


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": _nl(l)}
def code(*l): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _nl(l)}
def writefile(dest, path): return code(f"%%writefile {dest}\n" + path.read_text())
def _nl(lines):
    parts = "\n".join(lines).split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md("# w6 — GRASP planner: benchmark vs CEM & vanilla GD on Push-T",
       "",
       "Implements **GRASP** (Gradient-based Randomized Adaptive Search Planner)",
       "on top of a trained **LeWM**, and benchmarks it against **CEM** and",
       "**vanilla gradient descent** — measuring **success rate** and **wall-clock",
       "planning time** in the real Push-T env.",
       "",
       "Needs a trained LeWM checkpoint. Steps 1–3 reproduce the w4 setup so this",
       "notebook is self-contained; if you already trained LeWM in this session,",
       "skip to step 4. Runtime → **T4 GPU**."),

    md("## 0. GPU"),
    code("!nvidia-smi -L",
         "import torch; assert torch.cuda.is_available(), 'Enable a GPU runtime!'"),

    md("## 1. Install + clone (same as w4)"),
    code("%%bash",
         "set -e",
         "pip -q install 'stable-worldmodel[train,env]' zstandard imageio 2>&1 | tail -1",
         "apt-get -qq install -y xvfb zstd > /dev/null",
         "cd /content && [ -d le-wm ] || git clone -q https://github.com/lucas-maes/le-wm.git",
         "echo done"),
    code("import os, sys",
         "os.environ['STABLEWM_HOME'] = '/content/.stable-wm'",
         "os.makedirs(os.environ['STABLEWM_HOME'], exist_ok=True)",
         "os.chdir('/content/le-wm'); sys.path.insert(0, '/content/le-wm')"),

    md("## 2. Data (HDF5 from HF → Lance) — skip if already present"),
    code("import os",
         "home = os.environ['STABLEWM_HOME']",
         "h5 = os.path.join(home, 'pusht_expert_train.h5')",
         "lance = os.path.join(home, 'pusht_expert_train.lance')",
         "if not os.path.exists(lance):",
         "    from huggingface_hub import hf_hub_download",
         "    z = hf_hub_download('quentinll/lewm-pusht', 'pusht_expert_train.h5.zst',",
         "                        repo_type='dataset', local_dir=home)",
         "    os.system(f'zstd -d -f {z} -o {h5} && rm -f {z}')",
         "    from stable_worldmodel.data import convert",
         "    convert(h5, lance, dest_format='lance', mode='overwrite')",
         "print('data ready')"),

    md("## 3. Train LeWM from scratch — skip if checkpoint exists",
      "",
      "Same command as w4. ~1.5–3 h on a T4. If you saved a checkpoint from w4,",
      "upload it to `$STABLEWM_HOME/checkpoints/pusht/lewm/` and skip this."),
    code("%%bash",
         "cd /content/le-wm; export STABLEWM_HOME=/content/.stable-wm",
         f"CKPT=$STABLEWM_HOME/checkpoints/{CKPT}",
         "if [ -f \"$CKPT\" ]; then echo \"checkpoint exists, skipping training\"; else \\",
         "python train.py data=pusht \\",
         f"  img_size={IMG} trainer.max_epochs={EPOCHS} trainer.precision=16-mixed \\",
         f"  trainer.accelerator=gpu trainer.devices=1 loader.batch_size={BATCH} \\",
         f"  num_workers=2 output_model_name={RUN} ; fi"),

    md("## 4. Add the GRASP planner + benchmark scripts"),
    writefile("/content/le-wm/grasp_solver.py", HERE / "grasp_solver.py"),
    writefile("/content/le-wm/benchmark.py", HERE / "benchmark.py"),
    writefile("/content/le-wm/synthetic_benchmark.py", HERE / "synthetic_benchmark.py"),

    md("## 5. Sanity check: optimizer-level comparison (fast, CPU)",
      "",
      "Stresses the three optimizers on a rugged differentiable cost (a proxy for",
      "the world-model landscape) with matched compute budgets. Confirms GRASP",
      "beats budget-matched CEM and all GD variants, and is stable."),
    code("!python synthetic_benchmark.py"),

    md("## 6. Push-T benchmark: success rate + wall-clock planning time",
      "",
      "Closed-loop MPC in the real env, same start/goal pairs for every planner."),
    code("%%bash",
         "cd /content/le-wm; export STABLEWM_HOME=/content/.stable-wm",
         "xvfb-run -a python benchmark.py \\",
         f"  --model {CKPT} --img-size {IMG} --num-eval 20 \\",
         "  --planners cem,gd,grasp"),
    code("print(open('/content/le-wm/grasp_benchmark.md').read())"),

    md("## 7. Download results"),
    code("from google.colab import files",
         "import shutil, os",
         "os.makedirs('/content/w6_out', exist_ok=True)",
         "for f in ['grasp_benchmark.md','synthetic_results.md','grasp_solver.py','benchmark.py']:",
         "    p='/content/le-wm/'+f",
         "    if os.path.exists(p): shutil.copy(p,'/content/w6_out/')",
         "shutil.make_archive('/content/w6_out','zip','/content/w6_out')",
         "files.download('/content/w6_out.zip')"),

    md("## Notes",
      "",
      "- **GD budgets**: `gd` is literal vanilla single-start Adam (clipped); add",
      "  `gd_multi` to `--planners` for a stronger 32-start GD reference.",
      "- Wall-clock is measured around each `solver.solve()` with CUDA sync, so it",
      "  reflects real planning latency, not sampling budget.",
      "- Raise `--num-eval` for tighter success-rate estimates (slower)."),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = HERE / "GRASP_Benchmark_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "cells:", len(cells))
