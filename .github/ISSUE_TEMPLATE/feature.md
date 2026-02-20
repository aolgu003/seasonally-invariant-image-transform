---
name: Feature
about: Propose a new feature following the five-phase workflow
title: "feat: <short description>"
labels: feature
assignees: ''
---

## Phase tracker
- [ ] 1 Requirements — manager approves
- [ ] 2 Design — manager approves
- [ ] 3 BDD Specification — manager approves
- [ ] 4 Implementation — deviations logged below
- [ ] 5 Closeout — docs updated, issue closed

---

## 1. Requirements

### Problem statement
<!-- What user-visible problem does this solve? -->

### Acceptance criteria
<!-- Numbered, verifiable conditions that must all be true when done. -->
1.
2.

### Out of scope
<!-- Explicitly list what this issue will NOT address. -->

### Open questions
<!-- Surface ambiguities here. Do not make silent assumptions. -->
- [ ]

---

## 2. Design

### Approach
<!-- High-level description of the chosen solution. -->

### Affected files
| File | Change |
|------|--------|
|      |        |

### LaTeX sections to update
- [ ] `docs/sections/architecture.tex`
- [ ] `docs/sections/behaviors.tex`
- [ ] `docs/sections/algorithms.tex`
- [ ] `docs/sections/feature_index.tex` (always)

### Architecture diagram changes
<!-- Describe any mermaid diagram updates needed. -->

---

## 3. BDD Specification

### Behavior areas and scenarios

#### `TestArea`  →  `tests/path/test_file.py`

**Scenario: name**
- Given:
- When:
- Then:

---

## 4. Implementation notes

### Deviations from approved design
<!-- Log every deviation immediately. Do not work around deviations silently. -->
_None_

### Complexity flags
<!-- Stop and record unexpected complexity before proceeding. -->
_None_

---

## 5. Closeout

### Retrospective
<!-- What went well? What would you change next time? -->

### Docs updated
- [ ] LaTeX sections listed above
- [ ] `docs/sections/feature_index.tex` row added
- [ ] `CLAUDE.md` Current Context reset to none
