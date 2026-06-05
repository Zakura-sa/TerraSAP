# So2Sat / PaviaU / GRSS2013 Integration Notes

This document lists every source file that must be updated so TerraSAP can run FSCIL experiments on the three requested hyperspectral datasets without directly changing the code in this workspace. Follow each checklist item verbatim when editing your local working tree.

---

## 1. Dataset preparation (already scripted)

| Dataset | Script | Default output | Command |
| --- | --- | --- | --- |
| So2Sat-LCZ42 | `scripts/prepare_so2sat.py` | `data/So2Sat-LCZ42_ImageFolder` | `python scripts/prepare_so2sat.py --src data/So2Sat-LCZ42 --dst data/So2Sat-LCZ42_ImageFolder` |
| Pavia University | `scripts/prepare_paviau.py` | `data/PaviaU_ImageFolder` | `python scripts/prepare_paviau.py --src data/PaviaU --dst data/PaviaU_ImageFolder` |
| Houston 2013 (GRSS Data Fusion) | `scripts/prepare_grss2013.py` | `data/GRSS2013_ImageFolder` | `python scripts/prepare_grss2013.py --src data/GRSS2013 --dst data/GRSS2013_ImageFolder` |

Each script performs PCA (top 3 components), extracts `64×64` patches centered on labeled pixels, upsamples them to `224×224`, and records the FSCIL session plan inside the generated `metadata_*.json`.

---

## 2. Code changes required in TerraSAP

### 2.1 `utils/data.py`

Add three new dataset wrappers at the bottom of the file (after `class MSTAR`):

```python
class So2SatLCZ42(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [transforms.ToTensor()]
        # Base7 + 5×2 incremental order (matches prepare_so2sat.CLASS_ORDER)
        self.class_order = list(range(17))

    def download_data(self):
        train_dir = "./data/So2Sat-LCZ42_ImageFolder/train"
        test_dir = "./data/So2Sat-LCZ42_ImageFolder/test"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
```

```python
class PaviaUniversity(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [transforms.ToTensor()]
        # Session 0 (5 classes) + 4 incremental sessions
        self.class_order = [1, 2, 3, 4, 5, 0, 6, 7, 8]  # order defined by session plan

    def download_data(self):
        train_dir = "./data/PaviaU_ImageFolder/train"
        test_dir = "./data/PaviaU_ImageFolder/test"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
```

```python
class Houston2013(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [transforms.ToTensor()]
        # Base7 + four 2-class incremental sessions
        self.class_order = [0, 1, 2, 3, 4, 5, 14, 6, 7, 8, 9, 10, 11, 12, 13]

    def download_data(self):
        train_dir = "./data/GRSS2013_ImageFolder/train"
        test_dir = "./data/GRSS2013_ImageFolder/test"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
```

> Update the `class_order` arrays if you renumber the sub-folders. The indices above follow the natural alphabetical order described in each metadata file.

### 2.2 `utils/data_manager.py`

Extend `_get_idata` to dispatch the new dataset names:

```python
    elif name == "so2sat":
        return So2SatLCZ42(args)
    elif name == "paviau":
        return PaviaUniversity(args)
    elif name == "grss2013":
        return Houston2013(args)
```

Import the new wrappers at the top of the file:

```python
from utils.data import ..., NWPU_RESISC45, MSTAR, So2SatLCZ42, PaviaUniversity, Houston2013
```

### 2.3 Experiment configs (`exps/.../*.json`)

Create three experiment configs mirroring `exps/simplified_ablation/so2sat_full_system.json` but pointing to the right dataset:

* `exps/simplified_ablation/paviau_full_system.json`
  * `"dataset": "paviau"`
  * `"init_cls": 5`, `"increment": 1`, `"kshot": 5`
  * Reduce batch size to 16 if GPU memory is limited (patch count is smaller).

* `exps/simplified_ablation/grss2013_full_system.json`
  * `"dataset": "grss2013"`
  * `"init_cls": 7`, `"increment": 2`, `"kshot": 5`
  * Use the same optimizer / EMA knobs as NWPU/UCM.

* `exps/simplified_ablation/so2sat_full_system.json`
  * Already exists; simply ensure `"dataset": "so2sat"` and the new dataset class can load data.

### 2.4 Logging folders

The new experiments will write to:

```
logs/{model_name}/{paviau|grss2013|so2sat}/{init_cls}/{increment}/{kshot}/...
saved_model/{model_name}/{paviau|grss2013}/{init_cls}_{increment}/...
```

No extra code is necessary once the dataset names are recognized.

---

## 3. Session definitions

| Dataset | Session 0 (base) | Incremental sequence |
| --- | --- | --- |
| So2Sat-LCZ42 | Dense trees, Scattered trees, Low plants, Water, Compact HR, Open HR, Large LR | S1: {Compact MR, Open MR}; S2: {Compact LR, Open LR}; S3: {Sparsely built, Heavy industry}; S4: {Bush/scrub, Bare rock/paved}; S5: {Bare soil/sand, Lightweight LR} |
| Pavia University | Meadows, Trees, Bare soil, Shadows, Painted metal sheets | S1: Asphalt; S2: Bitumen; S3: Gravel; S4: Self-blocking bricks |
| Houston 2013 | Grass healthy, Grass stressed, Grass synthetic, Tree, Soil, Water, Running track | S1: Residential + Commercial; S2: Road + Highway; S3: Railway + Parking lot 1; S4: Tennis court + Parking lot 2 *(Parking lot 2 acts as the requested “Synthetic Track” class because the dataset does not provide a separate label.)* |

The orders above match the metadata files generated by the preprocessing scripts.

---

## 4. Training commands

```bash
python trainer.py --config exps/simplified_ablation/so2sat_full_system.json
python trainer.py --config exps/simplified_ablation/paviau_full_system.json
python trainer.py --config exps/simplified_ablation/grss2013_full_system.json
```

> Adjust `init_cls` / `increment` in each JSON if you run ablations (e.g., 7+2 for Houston, 5+1 for Pavia). The k-shot control in `DataManager` will sub-sample incremental classes to 5 shots when `kshot` is set.

---

## 5. Reviewer-4 linkage

* **Urban optical diversity** is covered by So2Sat (global urban LCZ patches), Pavia (dense European campus), and Houston (complex U.S. city scene).
* **Spectral modality diversity** now spans RGB (NWPU/UCM), SAR (MSTAR), and hyperspectral optical (Pavia/Houston/So2Sat).
* Preprocessing choices (PCA+patch upsampling, sliding-window centers) are aligned with the reviewer’s request; cite the metadata files inside each processed dataset when writing the rebuttal.

Once these edits are applied locally, TerraSAP can load the new datasets without touching the training script again.
