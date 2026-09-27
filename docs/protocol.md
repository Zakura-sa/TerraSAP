# Experiment protocol

Source: [TerraSAP, IEEE JSTARS](https://doi.org/10.1109/JSTARS.2025.3644602), Sections IV-A/B. The three `configs/` presets encode the following training schedule; implementation differences are listed separately in [implementation.md](implementation.md).

| Dataset | Base classes | Incremental sessions | Classes per increment | Shots per new class |
| --- | ---: | ---: | ---: | ---: |
| NWPU-RESISC45 | 25 | 4 | 5 | 5 |
| UCM | 12 | 3 | 3 | 5 |
| MSTAR | 4 | 6 | 1 | 5 |

NWPU/UCM use the RSSC protocol and MSTAR uses CPL. Evaluation covers all classes seen so far. Average accuracy is the arithmetic mean of session Top-1 accuracies, including the base session. Harmonic accuracy uses sample-level accuracies of previous-session classes and newly introduced classes; the base session is reported as N/A.

The paper reports Average accuracies of 84.52%, 84.41%, and 76.70% for NWPU, UCM, and MSTAR respectively. These are published results, not measurements from the checks in this repository.

## Training

| Setting | Value |
| --- | --- |
| Backbone | Frozen ViT-B/16, ImageNet-21k pretrained |
| Base training | 40 epochs |
| Optimizer | SGD, momentum 0.9, weight decay 0.0005 |
| Learning rate | 0.007, cosine decay |
| Batch size | 16 |
| Prompts | Six tokens, zero-initialized TIP |
| Training transforms | RandomResizedCrop(224), horizontal flip p=0.5 |
| Test transforms | Resize(256), CenterCrop(224) |
| Incremental sessions | Prototype/memory updates without iterative optimization |

The new presets enable these transforms through `paper_protocol: true`, disable layered learning rates and KL loss, and use RGB conversion and ImageNet normalization. Legacy experiment configs retain their original budgets and transforms.

## Components

- **SAPM:** two-dimensional positional encoding, relative position bias, and spatial/semantic/original prompt branches.
- **NCAC:** support-set prototypes combined with spatial context.
- **DPEMA:** separate original/modulated prompt EMA interfaces and fixed-weight fusion.
- **Anchor loss:** the paper selects a sample closest to each class mean within the current batch, alongside cross-entropy.

The publication also includes Pavia University and Houston 2013. Their retained preparation/configuration paths and unresolved session-plan differences are described in [datasets.md](datasets.md).

## Split references

- W. Wang et al., “Gradient guided multiscale feature collaboration networks for few-shot class-incremental remote sensing scene classification,” IEEE TGRS 62, 2024, Art. 5611912.
- Y. Zhao et al., “Few-shot class-incremental SAR target recognition via cosine prototype learning,” IEEE TGRS 61, 2023, Art. 5212718.
