# Episode Cleaning and 3D Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely quarantine 8-11 episodes shorter than 5 seconds and provide a portable browser tool for inspecting uploaded CSV trajectories in interactive 3D.

**Architecture:** A focused Python module owns episode inspection, rejection decisions, safe moves, and manifests; a thin CLI exposes dry-run and apply modes. A standalone HTML page parses CSV locally and uses Plotly WebGL for interactive 3D rendering.

**Tech Stack:** Python 3, pytest, standard library CSV/JSON/pathlib, HTML/CSS/JavaScript, Papa Parse, Plotly.js.

## Global Constraints

- Reject only episodes with duration strictly less than 5 seconds.
- Move rejected files to `data/factory/8-11/_rejected_short/<task>/`; never delete or overwrite.
- Exclude quarantine and generated manifests from scans.
- Keep uploaded CSV contents local to the browser.
- Detect `right_controller`, `right_tcp`, `left_controller`, and `left_tcp` XYZ groups.

---

### Task 1: Episode inspection and quarantine rules

**Files:**
- Create: `scripts/clean_short_episodes.py`
- Create: `tests/test_clean_short_episodes.py`

**Interfaces:**
- Produces: `EpisodeInfo`, `inspect_episode(path)`, `scan_episodes(root)`, `is_short(info, threshold_s)`, `quarantine_path(info, root)`, and `clean(root, threshold_s, apply)`.

- [ ] **Step 1: Write failing tests** for first/last `t` duration, strict `<5` behavior, quarantine directory exclusion, and collision-safe destination naming.
- [ ] **Step 2: Run** `python -m pytest tests/test_clean_short_episodes.py -v`; expect missing-module failure.
- [ ] **Step 3: Implement** streaming CSV inspection, pure decision helpers, safe destination resolution, and move execution using `pathlib` and `shutil`.
- [ ] **Step 4: Run the focused tests** and expect all tests to pass.

### Task 2: CLI and audit manifests

**Files:**
- Modify: `scripts/clean_short_episodes.py`
- Modify: `tests/test_clean_short_episodes.py`

**Interfaces:**
- Consumes: Task 1 inspection and cleaning functions.
- Produces: CLI options `ROOT`, `--threshold`, `--apply`, `--manifest-prefix`; CSV and JSON records with source, destination, rows, duration, threshold, and action.

- [ ] **Step 1: Add failing tests** proving dry-run leaves files in place and apply mode moves files while writing equivalent CSV/JSON manifests.
- [ ] **Step 2: Run focused tests** and confirm manifest assertions fail.
- [ ] **Step 3: Implement** deterministic manifest serialization and an argparse entry point with dry-run default.
- [ ] **Step 4: Run focused tests** and expect all tests to pass.

### Task 3: Standalone 3D viewer

**Files:**
- Create: `tools/episode-3d-viewer/index.html`
- Create: `tools/episode-3d-viewer/README.md`
- Create: `tests/test_episode_3d_viewer.py`

**Interfaces:**
- Consumes: uploaded CSV files with `t`, optional `coordinate_frame`, and any supported XYZ group.
- Produces: drag/drop and file-picker UI, local parsing, Plotly 3D traces, per-file summary, errors, legend toggles, equal axes, and camera reset.

- [ ] **Step 1: Write failing structural tests** for local file input, drop zone, supported column prefixes, parsing/rendering libraries, equal aspect mode, and accessible status/error regions.
- [ ] **Step 2: Run** `python -m pytest tests/test_episode_3d_viewer.py -v`; expect missing-file failure.
- [ ] **Step 3: Implement the page** with semantic HTML, responsive CSS, keyboard-accessible controls, Papa Parse row validation, bounded rendering downsampling, and Plotly `scatter3d` traces.
- [ ] **Step 4: Add usage documentation** for double-click/local-server use, controls, supported columns, privacy, and error messages.
- [ ] **Step 5: Run structural tests** and expect all tests to pass.

### Task 4: Apply cleaning and end-to-end verification

**Files:**
- Create: `data/factory/8-11/cleaning_manifest_20260811.csv`
- Create: `data/factory/8-11/cleaning_manifest_20260811.json`
- Move: two CSV files shorter than 5 seconds into `_rejected_short/<task>/`.

**Interfaces:**
- Consumes: Tasks 1–3 deliverables.
- Produces: cleaned active dataset, recoverable quarantine, audit artifacts, verified viewer.

- [ ] **Step 1: Dry-run** `python scripts/clean_short_episodes.py data/factory/8-11 --threshold 5 --manifest-prefix data/factory/8-11/cleaning_manifest_20260811`; expect exactly two `would_quarantine` records.
- [ ] **Step 2: Apply** the same command with `--apply`; expect exactly two moved files and no overwrite.
- [ ] **Step 3: Re-run dry-run**; expect zero active episodes below 5 seconds.
- [ ] **Step 4: Run** `python -m pytest tests/test_clean_short_episodes.py tests/test_episode_3d_viewer.py -v`; expect all tests to pass.
- [ ] **Step 5: Validate manifests** against quarantine paths and report the final active/rejected counts.

