"""Generate the turnkey Colab notebook LeWM_PushT_Colab.ipynb.

Embeds the local lewm_extras/*.py as %%writefile cells so the notebook is fully
self-contained (no manual uploads needed).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = HERE / "lewm_extras"

# ---- config knobs surfaced at the top of the notebook (free-T4 friendly) ----
IMG = 112
EPOCHS = 20
BATCH = 96
RUN = "pusht/lewm"          # checkpoints/pusht/lewm/... ; eval finds it here
EPOCH_CKPT = f"{RUN}/weights_epoch_{EPOCHS}.pt"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(_nl(lines))}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(_nl(lines))}


def writefile(dest, path):
    body = path.read_text()
    return code(f"%%writefile {dest}\n" + body)


def _nl(lines):
    joined = "\n".join(lines)
    parts = joined.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md("# LeWorldModel (LeWM) on Push-T — from-scratch replication",
       "",
       "Runs on a **free Colab T4 GPU**. Trains LeWM from scratch on the Push-T",
       "expert dataset (downloaded as **HDF5** from HuggingFace), validates the",
       "success rate with CEM planning in the real Push-T env, then produces",
       "decoded dream rollouts and a t-SNE of the latent space.",
       "",
       "> Runtime → Change runtime type → **T4 GPU** before running.",
       "",
       "Notes on the free-tier budget are in the final cell; defaults",
       f"(img={IMG}, epochs={EPOCHS}, batch={BATCH}) are tuned to finish inside one",
       "session. Scale them up toward the paper config on a bigger GPU."),

    md("## 0. Check the GPU (must be CUDA — the repo is CUDA-only)"),
    code("!nvidia-smi -L",
         "import torch; assert torch.cuda.is_available(), 'Enable a GPU runtime!'",
         "print('CUDA device:', torch.cuda.get_device_name(0))",
         "# T4 is Turing -> supports fp16 but NOT bf16; we use 16-mixed below."),

    md("## 1. Install LeWM + stable-worldmodel and clone the paper repo"),
    code("%%bash",
         "set -e",
         "pip -q install 'stable-worldmodel[train,env]' zstandard imageio 2>&1 | tail -2",
         "# Push-T is a pymunk/pygame env -> needs a headless framebuffer for eval;",
         "# zstd decompresses the dataset archive.",
         "apt-get -qq install -y xvfb zstd > /dev/null",
         "cd /content && rm -rf le-wm && git clone -q https://github.com/lucas-maes/le-wm.git",
         "echo 'installed.'"),
    code("import os, sys",
         "os.environ['STABLEWM_HOME'] = '/content/.stable-wm'",
         "os.makedirs(os.environ['STABLEWM_HOME'], exist_ok=True)",
         "os.chdir('/content/le-wm'); sys.path.insert(0, '/content/le-wm')",
         "print('STABLEWM_HOME =', os.environ['STABLEWM_HOME'])"),

    md("## 2. Download the Push-T **HDF5** dataset from HuggingFace",
      "",
      "`quentinll/lewm-pusht` ships `pusht_expert_train.h5.zst` (~13 GB). We",
      "decompress it to `$STABLEWM_HOME` (used directly by evaluation), then",
      "convert a copy to the compact Lance table (~0.8 GB) that training reads",
      "for fast random access."),
    code("# download the HDF5 archive (~13 GB) via the stable hub API",
         "import os",
         "from huggingface_hub import hf_hub_download",
         "home = os.environ['STABLEWM_HOME']",
         "p = hf_hub_download('quentinll/lewm-pusht', 'pusht_expert_train.h5.zst',",
         "                    repo_type='dataset', local_dir=home)",
         "print('downloaded ->', p)"),
    code("%%bash",
         "set -e",
         "cd $STABLEWM_HOME",
         "echo 'decompressing (a few minutes)...'",
         "zstd -d -f pusht_expert_train.h5.zst -o pusht_expert_train.h5",
         "rm -f pusht_expert_train.h5.zst   # reclaim disk",
         "ls -lh pusht_expert_train.h5"),
    code("# HDF5 -> Lance (fast training format). Keeps the .h5 for eval.",
         "import os",
         "from stable_worldmodel.data import convert",
         "src = os.path.join(os.environ['STABLEWM_HOME'], 'pusht_expert_train.h5')",
         "dst = os.path.join(os.environ['STABLEWM_HOME'], 'pusht_expert_train.lance')",
         "convert(src, dst, dest_format='lance', mode='overwrite')",
         "print('converted ->', dst)"),
    code("!swm datasets   # sanity-check: should list pusht_expert_train (Lance + HDF5)"),

    md("## 3. Train LeWM **from scratch** on Push-T",
      "",
      "Uses the paper repo's `train.py` (its real `jepa.py` model + SIGReg +",
      "next-embedding loss). Overrides adapt the CUDA/bf16 defaults to a free T4",
      "and a single-session budget. This is genuine from-scratch training —",
      "`encoder.pretrained=false` in the model config."),
    code("%%bash",
         "cd /content/le-wm",
         f"export STABLEWM_HOME=/content/.stable-wm",
         "python train.py \\",
         "  data=pusht \\",
         f"  img_size={IMG} \\",
         f"  trainer.max_epochs={EPOCHS} \\",
         "  trainer.precision=16-mixed \\",
         "  trainer.accelerator=gpu trainer.devices=1 \\",
         f"  loader.batch_size={BATCH} \\",
         "  num_workers=2 \\",
         f"  output_model_name={RUN}"),
    code("!swm checkpoints pusht   # confirm weights_epoch_*.pt were written"),

    md("## 4. Validate the success rate (CEM planning in the real Push-T env)",
      "",
      "Replays expert start/goal pairs and plans with the trained world model.",
      "`metrics` reports the **success rate** (fraction of episodes solved).",
      "Run under `xvfb` because Push-T renders through pygame."),
    code("%%bash",
         "cd /content/le-wm",
         "export STABLEWM_HOME=/content/.stable-wm",
         "xvfb-run -a python eval.py --config-name=pusht.yaml \\",
         f"  policy={EPOCH_CKPT} \\",
         f"  eval.img_size={IMG} \\",
         "  eval.num_eval=20 \\",
         "  solver.device=cuda",
         "# success rate is printed to stdout above as `metrics`, and appended to:",
         "echo '--- results file ---'; tail -n 20 $STABLEWM_HOME/pusht/lewm/pusht_results.txt 2>/dev/null || true"),

    md("## 5. Save the visualization scripts",
      "",
      "The paper's JEPA has no decoder, so the decoded dream rollouts use a",
      "small pixel decoder trained *after* training on the **frozen** latents",
      "(a probe — it never touches the world-model weights)."),
    writefile("/content/le-wm/decoder.py", EX / "decoder.py"),
    writefile("/content/le-wm/lewm_common.py", EX / "lewm_common.py"),
    writefile("/content/le-wm/train_decoder.py", EX / "train_decoder.py"),
    writefile("/content/le-wm/dream_rollout.py", EX / "dream_rollout.py"),
    writefile("/content/le-wm/tsne_latents.py", EX / "tsne_latents.py"),

    md("## 6. Train the probe decoder (frozen world model)"),
    code("%%bash",
         "cd /content/le-wm; export STABLEWM_HOME=/content/.stable-wm",
         f"python train_decoder.py --model {EPOCH_CKPT} \\",
         f"  --img-size {IMG} --epochs 8 --out /content/decoder_pusht.pt"),

    md("## 7. Decoded dream rollouts",
      "",
      "Feed 3 context frames, dream future latents open-loop under the real",
      "action sequence, and decode. Top row = ground truth, bottom = imagined."),
    code("%%bash",
         "cd /content/le-wm; export STABLEWM_HOME=/content/.stable-wm",
         f"python dream_rollout.py --model {EPOCH_CKPT} \\",
         f"  --decoder /content/decoder_pusht.pt --img-size {IMG} \\",
         "  --horizon 8 --episodes 3 --out /content/dream.png"),
    code("from IPython.display import Image, display",
         "display(Image('/content/dream.png'))",
         "display(Image('/content/dream.gif'))"),

    md("## 8. t-SNE of the latent space",
      "",
      "Structure by block angle / positions is evidence the latents encode",
      "physical state (the probing result from the paper)."),
    code("%%bash",
         "cd /content/le-wm; export STABLEWM_HOME=/content/.stable-wm",
         f"python tsne_latents.py --model {EPOCH_CKPT} \\",
         f"  --img-size {IMG} --num 3000 --out /content/tsne.png"),
    code("from IPython.display import Image, display",
         "display(Image('/content/tsne.png'))"),

    md("## 9. Bundle the deliverables"),
    code("%%bash",
         "cd /content",
         "mkdir -p lewm_deliverables",
         "cp -f dream.png dream.gif tsne.png lewm_deliverables/ 2>/dev/null || true",
         "cp -f .stable-wm/pusht/lewm/pusht_results.txt lewm_deliverables/ 2>/dev/null || true",
         "cp -rf .stable-wm/pusht/lewm/*.mp4 lewm_deliverables/ 2>/dev/null || true",
         "cp -f .stable-wm/checkpoints/pusht/lewm/config.json lewm_deliverables/ 2>/dev/null || true",
         "zip -qr lewm_deliverables.zip lewm_deliverables; ls -lh lewm_deliverables.zip"),
    code("from google.colab import files; files.download('/content/lewm_deliverables.zip')"),

    md("## Free-tier budget notes",
      "",
      f"- Defaults: img={IMG}px, {EPOCHS} epochs, batch {BATCH}, fp16. Reaches a",
      "  useful (not paper-maximal) success rate inside one free session.",
      "- **Disk**: the decompressed `.h5` is large (tens of GB). Standard Colab",
      "  disk (~100 GB) is enough; if you hit a limit, delete the `.h5` after",
      "  the Lance convert and skip step 4 (eval needs the `.h5`).",
      "- To approach the paper: `img_size=224`, `trainer.max_epochs=100`,",
      "  `loader.batch_size=128`, on an A100/H200 (bf16). Everything else is the",
      "  same command."),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "LeWM_PushT_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "cells:", len(cells))
