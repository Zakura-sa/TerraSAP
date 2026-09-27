# Class and sample inventories

Each dataset supplies a JSON class/session manifest and a CSV image-path inventory. Paths start with `data/` and are relative to the execution directory; images are obtained separately. These files describe reference layouts and are not automatically enforced by the ImageFolder loader. Adapt the prefix when using a different `data_root`.

Session 0 is the base session (paper S1). Evaluation accumulates all seen classes.

| Dataset | Classes per session | Train | Test | Per-class counts |
| --- | --- | ---: | ---: | --- |
| NWPU | 25,5,5,5,5 | 22455 | 9045 | 499 train / 201 test |
| UCM | 12,3,3,3 | 765 | 1335 | Base: 60/40; incremental: 5/95 |
| MSTAR | 4,1,1,1,1,1,1 | 1093 | 2426 | See JSON |

NWPU/UCM follow the retained ImageFolder alphabetical order. MSTAR uses:

```text
Base: BTR70, 2S1, BRDM2, BMP2
New:  ZIL131 -> T62 -> D7 -> BTR60 -> T72 -> ZSU234
```

Its experiment-to-ImageFolder mapping is `[4,0,2,1,8,6,5,3,7,9]`.

UCM/MSTAR incremental training folders contain five images per class. The NWPU inventory describes the candidate pool; each run samples five supports per new class and reuses them across standard training, NCAC, and EMA views. Actual support paths are logged.

Some class names in the paper's NWPU S2 analysis differ from this alphabetical order. The final historical ordering and original five-shot sample identities remain unverified. Renamed MSTAR files do not include an original SAR chip/serial mapping.

## Fields

- `imagefolder_id`: alphabetical folder label.
- `experiment_id`: model class label.
- `introduced_session`: session that introduces the class.
- `path`: relative image path.
- `bytes`: file size when inventoried, not a content hash.
