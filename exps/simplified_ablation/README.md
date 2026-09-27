# Legacy ablation and extension configurations

These presets retain the existing experiment paths, budgets, seed lists, and augmentation options. For the paper-schedule NWPU/UCM/MSTAR presets, use [`configs/`](../../configs/) instead.

NWPU, UCM, MSTAR, and SIRI-WHU have four configuration families:

| Suffix | Intended variant |
| --- | --- |
| `baseline` | Original prompt baseline |
| `efficiency_optimization` | Feature-modulation variant |
| `efficiency_plus_newclass` | Modulation and new-class-aware classification |
| `full_system` | Spatial prompts, NCAC, spatial-context and EMA interfaces |

Inspect the actual flags in each JSON; names do not guarantee that every module has an independent effect. In particular, the current default evaluation path disables modulation. See [implementation notes](../../docs/implementation.md).

Run from the repository root after preparing the data and adjusting the device, seed, and weights:

```bash
python main.py --config exps/simplified_ablation/nwpu_baseline.json
python main.py --config exps/simplified_ablation/ucmerced_full_system.json
python main.py --config exps/simplified_ablation/mstar_full_system.json
```

Without `pretrained_model_name`, these legacy configs retain timm's default checkpoint resolution and emit a warning. Set a verified checkpoint identifier explicitly for a controlled comparison. No config filename identifies the historical paper weights.

Pavia University, Houston 2013, and So2Sat also have `*_full_system.json` presets. Their preparation/session-plan differences are documented in [dataset notes](../../docs/datasets.md). Existing budgets are intentionally preserved and may differ from the new 40-epoch presets.

Logs are written under `logs/`, and final-epoch base checkpoints under `saved_model/`. Incremental memory is not a complete serialized recovery state. Reusing a base checkpoint requires `resume: true`; use distinct prefixes for independent experiments.
