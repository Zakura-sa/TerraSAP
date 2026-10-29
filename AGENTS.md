# Repository guidance

## Start here

Read [README.md](README.md), [repository status](docs/repository-status.md), [protocol](docs/protocol.md), [implementation notes](docs/implementation.md), and [dataset caveats](docs/datasets.md) before changing experiment-facing behavior.

- Record the actual checkout path, branch, remote, and staged/unstaged changes before editing. Recheck live refs when reasoning about delivery status.
- `main` is the integrated code line. `repro/paper-protocol-20260927` is the retained PR #1 snapshot, already merged into `main`. Documentation follow-ups target `main`; do not present that snapshot as an unmerged or independent enhancement.
- Compare both raw diffs and `git diff --ignore-cr-at-eol` before selecting changes. CRLF-only differences are not scientific fixes. Do not bulk-normalize line endings.
- Preserve unrelated dirty files and other local copies. Do not reset, stash, clean, or overwrite them. Inspect any relevant existing patch by path; never bundle an entire dirty checkout.

## Scientific boundaries

- Distinguish published results, reconstructed source, protocol presets, mocked/static checks, and newly measured results. Engineering checks do not validate or overturn historical scientific conclusions.
- Follow `research_owner`'s current queue and authorized protocol. Continue routine fixes, regression checks, and already authorized steps without seeking repeated approval for unchanged scope.
- Preserve existing method definitions, seed lists, training budgets, class/session plans, and reported results. Changes to the research question, data, budget, or evaluation protocol (including metrics) go to `research_owner` for a decision. Keep documented SAPM/NCAC/DPEMA and checkpoint/split limitations visible.
- Use lightweight CPU checks for maintenance unless the current queue already authorizes model/weight loading, training, experiments, GPU validation, or data downloads.
- Escalate unresolved provenance, complex cross-interface changes, missing credentials, and scope/API decisions rather than guessing.

## Lightweight validation and delivery

Run the dependency-free CPU suite from the repository root:

```bash
python3 -B -m unittest discover -s tests -v
```

For documentation/configuration work, also check relative Markdown links, parse the relevant JSON files with the standard library, and run `git diff --check` on the selected patch. Do not install model dependencies merely to run these checks. Report their limited scope and any remaining gaps.

Stage explicit paths only; never use `git add -A`. Commit/push only the reviewed selection to a named repair branch, without force pushing. Read back the remote ref and report the commit and PR applicability. Do not merge/close PRs or change repository visibility automatically.

Research agents use `openai-codex/gpt-6.1-sol` with `thinking=max` and `fast=false`.

The executor exclusively manages `.pi/settings.json`, local model configuration, and research-role prompts. Leave these files to the executor.
