# TCP Desktop Animation Design

## Goal

Provide a native desktop window for loading one episode CSV and replaying both left and right TCP poses as moving tool coordinate frames over their 3D trajectories.

## Architecture

`scripts/episode_3d_desktop.py` contains a data layer independent of Qt (`TcpSeries`, CSV parsing, quaternion conversion, timeline lookup) and a PySide6 window containing an embedded Matplotlib 3D canvas. A `QTimer` advances playback against CSV `t` values at a display cadence near 30 FPS.

## Visualization and Controls

- Show both TCP paths simultaneously by default: right in orange and left in blue.
- At the current timestamp, show each TCP origin plus a red X, green Y, and blue Z tool-axis triad computed from the CSV `w,x,y,z` quaternion.
- Preserve equal XYZ scale and mouse rotation/zoom.
- Provide Open CSV, Play/Pause, time slider, current/total time, speed selection, loop toggle, and per-arm visibility toggles.
- Keep the full path faint and progressively emphasize the traversed portion.
- Scale tool axes from the loaded trajectory extent, with a nonzero fallback for stationary data.

## Timing and Data Rules

- Use the recorded `t` column as the authoritative timeline.
- Render at approximately 30 FPS while selecting the nearest recorded pose; do not rewrite or resample the CSV.
- Normalize valid quaternions before rotation conversion. Treat near-zero or non-finite quaternions as invalid rows.
- Allow either arm to be absent; reject a file only when neither complete TCP position/quaternion group has usable rows.

## Error Handling and Verification

- Show file and schema errors in a modal and keep the prior valid episode loaded.
- Unit-test quaternion axes, strict row validation, timeline lookup, and single-/dual-arm parsing.
- Run the desktop window headlessly for construction and sample loading, then launch it visibly for final validation.

