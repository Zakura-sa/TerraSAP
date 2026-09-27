# Implementation notes

The source retains frozen-backbone prompting, deterministic prompt generation, cosine classification, NCAC additive/multiplicative prototype mixing, and fixed-weight EMA fusion. The NCAC branch preserves initialized new-class weights instead of overwriting them with ordinary class means.

## Code map

| Behavior | Source |
| --- | --- |
| Frozen ViT and Deep prompt insertion/removal | `backbone/asp_backbone.py` |
| Position encoding and relative attention | `backbone/enhanced_prompts.py` |
| Spatial/semantic/original modulation branches | `backbone/lightweight_feature_modulation.py` |
| Context-aware new-class initialization | `backbone/new_class_aware_classifier.py` |
| Classifier preservation and EMA updates | `models/asp.py` |
| Fixed incremental supports | `utils/support.py`, `utils/data_manager.py` |
| Grouped accuracy and session-level harmonic score | `utils/metrics.py`, `trainer.py` |

## Runtime conventions

- `configs/` contains three paper-schedule presets. Existing `exps/` presets and extension loaders remain available with their historical hyperparameters.
- Standard incremental training views share one seeded support selection per class across NCAC, training, and EMA/prototype extraction. Test data remain complete. Legacy sample-removal and validation-split helpers retain separate behavior.
- Grouped accuracy includes the maximum class label; old/new metrics use the current session boundary. Base-session harmonic accuracy is N/A.
- Training keeps final-epoch base weights and the trained base classifier. It does not select weights by test accuracy or delete other seeds' checkpoints. Resuming an existing base checkpoint requires `resume: true`.
- Explicit `pretrained_model_name` is used by both spatial and baseline prompt backbones. Null/empty values fail. An absent key preserves legacy timm resolution with a warning, not a verified ImageNet-21k identity.
- Seed lists, anchor weights, EMA coefficients, and internal NCAC constants retain existing values where the paper does not uniquely establish them.

## Remaining algorithm and provenance differences

| Item | Paper | Current implementation |
| --- | --- | --- |
| SAPM relative bias | Distance mapping weighted by spatial relation, including prompt–patch relations | Query-dependent relative keys/values; relation weight scales the MLP residual; patch inputs are mean-pooled |
| Modulation and DPEMA | Smooth positive branch weights and a distinct modulated stream | Absolute-valued weights and additionally pooled noise; evaluation disables modulation, so the two streams are identical on the default path and the distinct dual-path effect is not implemented |
| NCAC context | Spatial-prompt-derived class context | Intermediate encoder vectors; context projection lacks a direct training signal |
| Anchors | Selected within the current batch | Selected over the base training dataset and appended to batches |
| Saved state | Complete evaluation state | Base `state_dict` only; EMA tensors are not registered buffers, so this is not a complete incremental recovery checkpoint |
| Weights and splits | ImageNet-21k checkpoint and RSSC/CPL protocols | Historical checkpoint identity is unresolved; NWPU final class order/support IDs and original MSTAR chip identities need verification |

The retained adaptive-fusion utility is not wired into the main training path. Configuration fields alone do not establish an active algorithm variant. Extension preprocessing/session-plan issues are documented in [datasets.md](datasets.md).

## Validation scope

The standard-library tests cover training budgets, 37,119 indexed paths, class mappings, fixed support reuse, metrics, source syntax, and selected loader/checkpoint contracts using mocks. They do not establish full model import/forward compatibility, GPU behavior, or reproduced accuracy. Dependencies, exact weights, and complete training must be validated together before reporting experimental results.
