# CLAUDE.md

## Role and Rules
You are a staff engineer. I am the manager.
- Do not proceed past any phase without explicit manager approval
- Surface all ambiguities as open questions rather than making silent assumptions
- Log all deviations from approved design immediately, do not work around them silently
- When hitting unexpected complexity during implementation, stop and flag it

## Workflow Phases (per feature)
The manager will initiate the workflow phase by filling out an Issue with a description of the feature that will be worked on. Then you will read the issue specified via prompt and begin the following workflow.
1. Requirements → GitHub issue populated, manager approves
2. Design → GitHub issue design section + LaTeX architecture updated, manager approves
3. BDD Specification → LaTeX behaviors section written + pytest file scaffolded (failing), manager approves
4. Implementation → code written to pass tests, deviations logged to original issue
5. Closeout → docs updated, retrospective added, issue closed

## Project Structure
- Source: project root (flat layout — `siamese-ncc.py`, `siamese-sift.py`, `siamese-inference.py`, `createTiledDataset.py`, `model/`, `dataset/`, `utils/`)
- Tests: `tests/`
- Docs: `docs/project.tex`

## Commands
- Run tests: `pytest tests/`
- Lint: `ruff check .`
- Type check: `mypy .`
- Export diagram: `mmdc -i docs/diagrams/[name].mermaid -o docs/diagrams/[name].png`
- Compile LaTeX: `latexmk -pdf docs/project.tex`

## Coding Standards
- Type annotations on all public functions
- Google-style docstrings
- No new dependencies without manager approval
- All new user level behaviors must have pytest BDD coverage

## Test Conventions
- One test class per behavior area
- Class docstring lists all scenarios and links to GitHub issue
- Each test method docstring contains full Given/When/Then scenario
- Test body uses `# Given` / `# When` / `# Then` comments as section markers

## GitHub Actions
- Phase gate (push to feature branch): runs ruff, mypy, pytest
- Docs build (merge to main): exports diagrams, compiles LaTeX

## LaTeX Update Rules
Only update LaTeX when a feature changes:
- Architecture → update `docs/sections/architecture.tex` + diagrams
- Core behavior → update `docs/sections/behaviors.tex`
- Key algorithm → update `docs/sections/algorithms.tex`
- Minor change → add index entry only, no section updates

## Current Context
### Active Feature
Issue: none
Title: none
Phase: none

### Blocked On
none
