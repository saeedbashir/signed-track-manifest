# Contributing

Thank you for looking. This repository holds a format (`SPEC.md`, the schemas
in `src/stm/schemas/`), a validator, and a pipeline that produces manifests from
existing interpreted video. Contributions to any of the three are welcome; the
notes below are what will make yours quick to review.

## Two rules that are not negotiable

1. **Nothing here synthesises sign language, and nothing here may be used to
   pass off synthesised signing as human.** Every sign track carries a required
   `provenance` field with no default. A change that adds a default, relaxes the
   check, or generates signing will not be merged, however good the code.
2. **Machine-generated captions are always `reviewed: false`.** Only an explicit
   step naming a human reviewer can change that (`Caption.mark_reviewed`). A
   configuration flag that marks generated text as reviewed is the thing that
   rule exists to prevent.

## Before you open a pull request

```sh
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

All three must pass; CI runs the same commands. `mypy` is strict, on purpose.

## What a good change looks like

- **Format changes** go through `SPEC.md`, both schemas, `validate-stm`, the
  examples in `examples/` (a valid one and, where the rule can be broken, an
  invalid one), and a `CHANGELOG.md` entry under the next format version. Readers
  must keep accepting every earlier `stmVersion`. Open an issue first — the
  format is young and every field is a promise to players that do not exist yet.
- **Pipeline changes** come with a test that fails without them. The aligner's
  tests are the model: each is a small transcript, a few recognised words with
  times, and the cues that must come out. If the change was found by looking at
  real output, say what was seen, and on which title, in the commit message.
- **Measurements over impressions.** A detection or activity change is argued
  with the harness (`stm harness score`) against the ground truth in
  `harness/`, not with a screenshot. If it makes a number worse and you think it
  is still right, say so and why.
- **Dependencies** are added to `DEPENDENCIES.md` with the licence and the reason
  the standard library was not enough. No paid services, no calls out that a
  reader could not reproduce.
- **Content** is never committed. Media is fetched from publishers by
  `catalogue/sources.yaml`-style entries with a recorded acquisition route and
  licence; the hosts in `ingest.BLOCKED_HOSTS` are refused because their terms
  forbid downloading.

## Commit messages

The subject says what changed; the body says why, and what was measured. Look
at `git log` — the history is written so that the reasoning survives the
person who had it.

## Reporting a problem

Use the issue templates. For a manifest that should validate and does not (or
the reverse), attach the manifest and the validator's output. For a title the
pipeline gets wrong, the `analysis.json` it wrote and one frame are usually
enough to see what happened.

## Code of conduct

Everyone taking part is expected to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Licence

By contributing you agree that your contribution is licensed under the
repository's licence, Apache 2.0 (`LICENSE`).
