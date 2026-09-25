## Collaboration Preferences

- Tell it like it is; do not sugar-coat responses.
- Take a forward-thinking view.
- Prioritize practical outcomes.
- Be innovative and think outside the box.

## Project

- Specification: `docs/PROJECT_SPECIFICATION.md` is the source of truth. Changes to methodology go
  through the spec first and get a row in its decision log (§18).
- Documentation language: Czech. Code, identifiers and commit messages: English.
- Never commit data, run outputs or secrets. Sharadar data live in `data/` (git-ignored).
- The final holdout (after 2019-12-31) is protected by the OOS vault (spec §9.7). Do not read or
  evaluate strategies on holdout data outside a `final_evaluation` run.
- Every change that affects strategy selection is a new trial in the trial registry (spec §9.8).
