# Repository and reproduction status

Verified against GitHub on **2026-10-29 (UTC)**. This is a delivery/evidence map, not a new experiment report.

## Which branch to use

| Ref | Verified commit | Meaning |
| --- | --- | --- |
| GitHub `main` / fetched `origin/main` | `b68f289d6d3f3fe74a4a62774fde6ab57b253a00` | Integrated code; merge of PR #1 on 2026-09-27 |
| `repro/paper-protocol-20260927` | `6fbfe196606be49dc32f646fd4f09f03fc6d5f6f` | Retained PR #1 head; use only when pinning that integration snapshot |
| Pre-integration `main` | `9800367abe8a852bb12a5e92e6b61db7dc6ce4bf` | Reconstructed research release before the protocol/reproducibility patch |
| `repair/branch-delivery-docs-20261029` | Based on `b68f289` | Documentation-only follow-up for `main`; no new algorithm or reproduction result |

At these pinned commits, GitHub `main` and `repro` have the **same tree**, `f0f68ab4ab55edf3d5c9da1dcb7a843f7d52aa3c`. The integration is already delivered on `main`; no second merge of the reproduction branch is needed. A later documentation commit may change the tree without changing the experiment code.

The original checkout's local `main` still pointed at `9800367`; it was not advanced. Fetch refreshed `origin/main`, not that local branch. Use verified remote refs or pinned commits rather than assuming a local branch name proves delivery status.

### PR and source provenance

[PR #1](https://github.com/Zakura-sa/TerraSAP/pull/1) is closed **and merged**, not pending. Its base was `main` at `9800367`, its head was `repro/paper-protocol-20260927` at `6fbfe19`, and its merge commit is `b68f289`.

The actual base-to-head patch changes 43 files (+38,775 / -1,394 lines), including three paper-schedule presets, split inventories, tests/docs, and targeted support-sampling, metric, checkpoint-selection, final-epoch retention, and CPU-timing fixes. It is the **same protocol/reproducibility delivery** retained on `repro`, built on the pre-integration source, not a second independent method enhancement. The large insertion count includes the split CSV inventories; it is not a measure of algorithmic change.

The pre-integration README describes a reconstruction from an earlier saved backup after server files/checkpoints were lost. Neither the reconstructed release nor PR #1 establishes recovery of the original experimental snapshot. Legacy experiment presets and upstream ASP attribution remain in place.

For traceability only: the repository's all-state PR list contained only #1; `repos/Zakura-sa/TerraSAP/pulls/3` returned HTTP 404 and no `refs/pull/3/head` was advertised. No PR #3 base/head or patch could be verified here; do not infer another repository or classify an unavailable patch.

## Existing evidence, not reproduced accuracy

The following CPU-only checks passed again on the integrated code during this documentation repair:

- `python3 -B -m unittest discover -s tests -v`: **17 tests passed**. These cover paper budgets, 37,119 indexed paths/class mappings, seeded support reuse, metric boundaries, Python syntax, and selected loader/checkpoint/timing contracts with mocks.
- Standard-library JSON parsing: **25 files** under `configs/`, `exps/`, and `splits/` parsed successfully.
- Relative Markdown links were checked locally; the selected documentation patch passed `git diff --check`.

No training, full model import/forward, model/weight loading, GPU numerical validation, or dataset download was performed. The published average accuracies remain those in [protocol.md](protocol.md), not measurements from these checks.

### Still unresolved or not implemented

See [implementation.md](implementation.md) and [datasets.md](datasets.md) for the authoritative details:

- Historical ImageNet-21k checkpoint identity, dependency locking, NWPU class/support provenance, and original MSTAR chip identities remain unresolved.
- SAPM relative-bias behavior, NCAC context/training signal, and dataset-wide rather than within-batch anchor selection differ from the paper description.
- Default evaluation disables modulation, making the two EMA streams identical; the distinct dual-path effect is **not implemented** on that path. The adaptive-fusion utility is not wired into the main training path.
- Saved base weights are not a complete incremental recovery checkpoint; extension preprocessing/class orders/session plans still need reconciliation before scientific comparisons.

These limitations are preserved, not silently repaired or used to revise historical conclusions. Completing them or reporting accuracy requires a separately authorized scientific/provenance validation effort.

## Local diff hygiene

The original reproduction checkout was deliberately left untouched:

| Inspection | Observed result before this repair |
| --- | --- |
| `git diff --shortstat` | 49 files changed, 46,205 insertions(+), 46,205 deletions(-) |
| `git diff --ignore-cr-at-eol --exit-code` | No output; exit 0, no substantive changes |
| Read-only byte comparison to `HEAD` after CRLF-to-LF normalization | All 49 dirty paths equal |

Those line-ending differences are **not included** in the repair commit. Documentation was prepared in an isolated clean worktree based on fetched `origin/main`, without resetting, stashing, cleaning, or normalizing the original files.

## Recheck the mapping

```bash
git ls-remote origin refs/heads/main refs/heads/repro/paper-protocol-20260927 refs/pull/1/head refs/pull/3/head
git fetch --no-tags origin refs/heads/main:refs/remotes/origin/main refs/heads/repro/paper-protocol-20260927:refs/remotes/origin/repro/paper-protocol-20260927
git diff --exit-code 6fbfe196606be49dc32f646fd4f09f03fc6d5f6f b68f289d6d3f3fe74a4a62774fde6ab57b253a00
gh api repos/Zakura-sa/TerraSAP/pulls/1
gh api 'repos/Zakura-sa/TerraSAP/pulls?state=all&per_page=100'
```

The pinned diff is expected to be empty; live branch tips may advance. Reverify API state instead of treating branch names or an earlier report as proof. This repair does not merge/close PRs or alter repository visibility.
