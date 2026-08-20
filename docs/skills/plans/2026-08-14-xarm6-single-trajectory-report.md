# xArm6 Single-Trajectory PDF Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a concise 6–8 page Chinese static PDF that densely and accurately presents the final seal-bag and fold-box xArm6 trajectory-following videos, trajectories, failures, and metrics.

**Architecture:** A single focused Python generator reads authoritative JSON/NPZ/MP4 inputs, derives metrics and plots, writes traceable JSON/PNG intermediates, and assembles the final PDF with ReportLab. A small unittest module verifies quaternion/Yaw math, metric aggregation, failure-window extraction, and required outputs. PyMuPDF renders the completed PDF for page-count, text, and visual QA.

**Tech Stack:** Python 3, NumPy, Matplotlib, ReportLab, OpenCV, Pillow, PyMuPDF, unittest.

## Global Constraints

- Output is a static Chinese A4 landscape PDF of 6–8 pages.
- Keep prose short while showing detailed multi-panel plots and complete metrics.
- Do not modify source NPZ, JSON, provenance, or MP4 files.
- Do not describe single-trajectory frame success as repeated-trial success rate.
- Do not include next-round optimization recommendations.
- Use `C:\Windows\Fonts\msyh.ttc` for Chinese text.
- Write all outputs under `reports/factory_bimanual/xarm6_single_trajectory_optimization_report/`.

---

### Task 1: Derivation and validation primitives

**Files:**
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/generate_report.py`
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/test_generate_report.py`

**Interfaces:**
- Consumes: authoritative summary JSON and trajectory NPZ paths from the approved design.
- Produces: `quaternion_to_yaw_deg(q: np.ndarray) -> np.ndarray`, `contiguous_windows(mask: np.ndarray, time_s: np.ndarray) -> list[dict]`, `derive_run(label: str, summary_path: Path, trajectory_path: Path, video_path: Path) -> dict`.

- [ ] **Step 1: Write failing unit tests** for identity/+90° Yaw, a known boolean failure window, mean/P95/max position error, coverage, and source file existence.
- [ ] **Step 2: Run** `python -m unittest test_generate_report.py -v`; expect failures because the generator module does not yet exist.
- [ ] **Step 3: Implement the primitives** with normalized quaternions, `np.unwrap` for Yaw, boolean run-length extraction, finite-value checks, and summary-vs-NPZ comparison fields.
- [ ] **Step 4: Run** `python -m unittest test_generate_report.py -v`; expect all tests to pass.

### Task 2: Figures, video keyframes, and traceable metrics

**Files:**
- Modify: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/generate_report.py`
- Modify: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/test_generate_report.py`
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/derived_metrics.json`
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/figures/*.png`

**Interfaces:**
- Consumes: the `derive_run` dictionaries from Task 1.
- Produces: `extract_keyframes(video_path: Path, fractions: tuple[float, ...]) -> list[dict]`, `plot_trajectory_overview(run: dict, output: Path) -> None`, `plot_failure_detail(run: dict, output: Path) -> None`, and JSON-serializable derived metrics.

- [ ] **Step 1: Add failing tests** that request five ordered keyframes, verify timestamps are inside video duration, and assert the sealed-bag derived failure window covers 12.11–12.62 s.
- [ ] **Step 2: Run the focused tests** and confirm they fail before implementation.
- [ ] **Step 3: Implement frame extraction and plots**: fixed-view 3D target/actual XYZ plus XY/XZ projections, target/actual Yaw, position/orientation error, success state, XYZ failure deltas, joint-limit/singularity margins, retiming, candidate counts, and acceleration constraints.
- [ ] **Step 4: Write `derived_metrics.json`** with input paths, hashes/sizes, computed statistics, summary values, differences, failure windows, and video metadata.
- [ ] **Step 5: Run all unit tests** and check every expected PNG is non-empty.

### Task 3: Assemble the concise detailed PDF

**Files:**
- Modify: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/generate_report.py`
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/xarm6_single_trajectory_optimization_report.pdf`

**Interfaces:**
- Consumes: run dictionaries, keyframes, plots, and derived metrics from Tasks 1–2.
- Produces: `build_pdf(seal: dict, fold: dict, output_path: Path) -> None` and the final 6–8 page PDF.

- [ ] **Step 1: Add an output-contract test** that opens the PDF with PyMuPDF, asserts 6–8 pages, checks Chinese section titles are extractable, and rejects the phrase `下一轮优化建议`.
- [ ] **Step 2: Run the output-contract test** and confirm it fails while the PDF is absent.
- [ ] **Step 3: Implement seven A4-landscape pages**: overview, keyframes/provenance, seal trajectory/error, seal failure detail, fold trajectory/error, fold retiming/margins, and full metric table/objective conclusions.
- [ ] **Step 4: Run the generator** from `E:\YZH123123\single-arm-mount` and confirm it exits 0.
- [ ] **Step 5: Run all unit tests** and confirm the output contract passes.

### Task 4: Numerical and visual verification

**Files:**
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/qa_pages/page-*.png`
- Create: `single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/verification.json`

**Interfaces:**
- Consumes: final PDF, authoritative source files, and derived metrics.
- Produces: rendered page images and a machine-readable verification result.

- [ ] **Step 1: Recompute core statistics independently** and compare coverage, mean/max errors, durations, failure counts, and collision counts with summary JSON; require differences below `1e-9` for coverage and below display rounding for reported errors.
- [ ] **Step 2: Render every PDF page** at 150 dpi with PyMuPDF into `qa_pages/` and verify every image is non-empty and has identical landscape dimensions.
- [ ] **Step 3: Inspect the contact sheet** for missing glyphs, clipped axes, overlapping legends, unreadable labels, and blank panels; revise and regenerate if needed.
- [ ] **Step 4: Run final verification**: `python -m unittest test_generate_report.py -v` and `python generate_report.py --verify-only`; require exit code 0 and zero failed checks.
- [ ] **Step 5: Save `verification.json`** with page count, dimensions, source/output hashes, numeric checks, and generation timestamp.

## Execution Note

The workspace has no Git metadata, so commit steps are intentionally omitted. The user requested direct execution in the current task, which selects inline execution rather than subagent dispatch.
