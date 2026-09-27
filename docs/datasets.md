# Dataset preparation and extensions

Install the additional preparation dependencies:

```bash
pip install -r requirements-preparation.txt
```

Run preparation into fresh directories: existing outputs are not purged automatically. All loaders expect matching train/test class folders and accept a common `data_root`. Legacy sweep overrides `nwpu_data_root`, `ucm_data_root`, and `mstar_data_root` point directly to folders containing `train/` and `test/`.

## Input formats

| Script | Expected inputs | Conversion |
| --- | --- | --- |
| `prepare_paviau.py` | `PaviaU.mat` (`paviaU`), `PaviaU_gt.mat` (`paviaU_gt`) | Three-component PCA, labeled-pixel patches, RGB resizing |
| `prepare_grss2013.py` | `2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat` (`ans`), `train_roi.tif`, `val_roi.tif` | Three-component PCA, ROI-centered patches, RGB resizing |
| `prepare_so2sat.py` | `training.h5`, `validation.h5`, `testing.h5`, with `sen2` and `label` | Sentinel-2 B4/B3/B2 selection and resizing of 32×32 patches; not PCA |

The So2Sat training pool combines training and validation files; testing remains separate. Pavia computes PCA over the scene and randomly splits labeled pixels, so large neighboring patches can overlap across train/test. This is not a spatially disjoint evaluation protocol.

## Session plans

The published extension configs are preserved as historical experiment presets, not substituted for the three `configs/` protocols:

- Pavia: five base classes, then four one-class sessions.
- Houston: seven base classes, then four two-class sessions.
- So2Sat config: five base classes, then twelve one-class sessions. Its preparation script instead describes seven base classes and five two-class sessions.
- SIRI-WHU: six base classes, then three two-class sessions.

The Pavia/Houston scripts' named session plans do not match the loaders' numeric ImageFolder orders in every case. Moreover, extension presets with `shuffle: true` use a seed-dependent permutation instead of those fixed orders. Preparation metadata is not consumed by `DataManager`. Reconcile class names, shuffle settings, and session sizes for an experiment before comparing its results to the paper; this integration preserves rather than silently redefines those historical protocols.

## Auxiliary scripts

`scripts/sweep_param.py` generates configs from the retained NWPU/UCM/MSTAR ablation presets; run it from the repository root. Use `--dry_run` to generate configs without training. The main path uses fixed EMA fusion; adaptive-fusion experiments are not advertised as active variants.

The `HuiTu_*.py` scripts are retained figure/layout helpers with embedded data and, in some cases, external PDF/PNG inputs. They are not training-result validators. Optional dependencies are listed in `requirements-figures.txt`; PDF conversion additionally requires a system PDF renderer such as Poppler. These scripts have not been executed as part of the release checks.
