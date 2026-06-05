# TerraSAP

TerraSAP is a spatially aware prompt-based framework for few-shot class-incremental learning (FSCIL) in remote sensing image classification. The implementation builds on a frozen ViT-B/16 backbone with prompt tuning and adds three paper-facing components:

- Spatially Aware Prompt Module (SAPM)
- New Class-Aware Classifier (NCAC)
- Dual-Path Exponential Moving Average (DPEMA)

This repository is intended to contain source code, experiment configs, and data-preparation scripts only. Datasets, checkpoints, logs, t-SNE arrays, and generated paper artifacts are intentionally ignored by Git.

## Release Note

The trained deployment files and model checkpoints were originally kept on a server, but those files were not preserved correctly and are no longer available. This public version was reconstructed and completed from an earlier saved code backup. It is provided as a cleaned research-code release for reproducing and extending the TerraSAP experiments; pretrained or trained TerraSAP checkpoints are not included.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The paper experiments used PyTorch with CUDA on an NVIDIA GPU. CPU execution is only practical for lightweight import checks or small smoke tests.

## Data Layout

Prepare datasets under `data/` using ImageFolder-style train/test splits:

```text
data/
  NWPU-RESISC45/
    train/<class_name>/*
    test/<class_name>/*
  UCMerced_LandUse_processed/
    train/<class_name>/*
    test/<class_name>/*
  SIRI-WHU/
    train/<class_name>/*
    test/<class_name>/*
  MSTAR/
    train/<class_name>/*
    test/<class_name>/*
  So2Sat-LCZ42_ImageFolder/
    train/<class_name>/*
    test/<class_name>/*
  PaviaU_ImageFolder/
    train/<class_name>/*
    test/<class_name>/*
  GRSS2013_ImageFolder/
    train/<class_name>/*
    test/<class_name>/*
```

For hyperspectral and multimodal datasets, use the provided preparation scripts:

```bash
python scripts/prepare_paviau.py --src data/PaviaU --dst data/PaviaU_ImageFolder
python scripts/prepare_grss2013.py --src data/GRSS2013 --dst data/GRSS2013_ImageFolder
python scripts/prepare_so2sat.py --src data/So2Sat-LCZ42 --dst data/So2Sat-LCZ42_ImageFolder
```

The scripts reduce spectral bands to three PCA components, crop local patches around labeled pixels, resize them to 224 x 224, and write ImageFolder-compatible outputs.

## Running Experiments

Run a single config:

```bash
python main.py --config exps/simplified_ablation/nwpu_full_system.json
```

Other paper-facing configs include:

```bash
python main.py --config exps/simplified_ablation/ucmerced_full_system.json
python main.py --config exps/simplified_ablation/mstar_full_system.json
python main.py --config exps/simplified_ablation/paviau_full_system.json
python main.py --config exps/simplified_ablation/grss2013_full_system.json
```

The default configs use 5-shot incremental sessions. Base-session checkpoints are written to `saved_model/`, and logs are written to `logs/`.

## Main Configs

- `nwpu_full_system.json`: NWPU-RESISC45, 25 base classes + 4 sessions of 5 classes.
- `ucmerced_full_system.json`: UCMerced, 12 base classes + 3 sessions of 3 classes.
- `mstar_full_system.json`: MSTAR, 4 base classes + 6 sessions of 1 class.
- `paviau_full_system.json`: Pavia University, 5 base classes + 4 sessions of 1 class.
- `grss2013_full_system.json`: Houston 2013, 7 base classes + 4 sessions of 2 classes.

## Citation

If you use this code, cite the TerraSAP paper:

```bibtex
@article{zeng2026terrasap,
  title={TerraSAP: Spatially-Aware Prompt-based Framework for Few-Shot Class-Incremental Learning in Remote Sensing Image Classification},
  author={Zeng, et al.},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026}
}
```

## Acknowledgements

TerraSAP is built on top of the ASP codebase and research line. We sincerely thank the authors of **Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt (ASP, ECCV 2024)** for releasing their official implementation: [DawnLIU35/FSCIL-ASP](https://github.com/DawnLIU35/FSCIL-ASP). TerraSAP extends this prompt-based FSCIL foundation toward remote sensing scenarios with spatially aware prompts, new-class-aware classifier initialization, and dual-path prompt-memory updates.
