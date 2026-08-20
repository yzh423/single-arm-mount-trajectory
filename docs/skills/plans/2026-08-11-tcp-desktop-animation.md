# TCP Desktop Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native PySide6 window that animates both recorded TCP tool coordinate systems over their 3D trajectories.

**Architecture:** Pure NumPy/CSV functions parse and validate pose series and convert quaternions; a PySide6 window embeds a Matplotlib 3D canvas and advances time with a QTimer. UI rendering consumes only the tested data interfaces.

**Tech Stack:** Python 3.11, NumPy, PySide6, Matplotlib, pytest.

## Global Constraints

- Both TCPs are visible simultaneously by default.
- CSV `t` controls playback timing; display refresh targets 30 FPS.
- Quaternion field order is `w,x,y,z`; tool axes are X red, Y green, Z blue.
- Missing one arm is allowed; missing both is an error.
- The deliverable is a native window and makes no browser/network request.

---

### Task 1: Pose data model and quaternion math

**Files:**
- Create: `scripts/episode_3d_desktop.py`
- Create: `tests/test_episode_3d_desktop.py`

**Interfaces:**
- Produces: `TcpSeries`, `Episode`, `quaternion_to_matrix(quaternion)`, `load_episode(path)`, and `nearest_index(times, target)`.

- [ ] Write failing tests for identity and Z-rotation quaternions, invalid zero quaternion, nearest-time lookup, dual-arm parsing, and one-arm parsing.
- [ ] Run the focused tests and confirm missing-module failure.
- [ ] Implement immutable data classes, streaming-compatible CSV parsing through `csv.DictReader`, validation, and normalized quaternion math.
- [ ] Run focused tests and expect all data tests to pass.

### Task 2: Native window and animation controls

**Files:**
- Modify: `scripts/episode_3d_desktop.py`
- Modify: `tests/test_episode_3d_desktop.py`
- Create: `tools/episode-3d-desktop/README.md`
- Create: `tools/episode-3d-desktop/run_viewer.bat`

**Interfaces:**
- Consumes: Task 1 `Episode` and quaternion functions.
- Produces: `EpisodeViewerWindow`, `main()`, native file dialog, 3D canvas, timer playback, slider, speed/loop/arm controls.

- [ ] Add failing headless Qt tests for window construction, required controls, episode loading, frame advance, and simultaneous arm artists.
- [ ] Implement the focused PySide6 window and Matplotlib renderer.
- [ ] Add the launcher and concise usage documentation.
- [ ] Run headless Qt and complete focused tests.

### Task 3: Runtime verification

**Files:**
- No production file changes expected.

- [ ] Run `python -m pytest tests/test_episode_3d_desktop.py -v`.
- [ ] Construct the window with `QT_QPA_PLATFORM=offscreen`, load a real retained episode, advance frames, and verify both arm series and tool triads.
- [ ] Launch the visible window with a representative CSV for user use.

