# 8-11 Episode Cleaning and 3D Viewer Design

## Scope

- Clean CSV episodes under `data/factory/8-11` using a strict duration threshold: reject episodes whose final `t` minus initial `t` is less than 5 seconds.
- Preserve rejected files by moving them to `data/factory/8-11/_rejected_short/<task>/`.
- Record every decision in machine-readable CSV and JSON manifests.
- Provide a portable browser-based viewer that accepts one or more episode CSV files and renders their 3D trajectories.

## Architecture

The cleaning command is a small Python module with pure functions for reading episode metadata, deciding whether an episode is short, and building collision-safe destination paths. A CLI applies those decisions. Dry-run is the default; `--apply` performs moves. Manifests record source path, destination path, row count, duration, threshold, action, and timestamp.

The viewer is a standalone HTML application. Papa Parse reads CSV files locally in the browser and Plotly renders WebGL 3D lines. No CSV data is uploaded to a server. The page detects the four known position groups (`right_controller`, `right_tcp`, `left_controller`, `left_tcp`) by their `_pos_x`, `_pos_y`, and `_pos_z` columns.

## Viewer Behavior

- Accept drag-and-drop or file-picker upload for one or multiple CSV files.
- Draw each available position group as a distinct, labeled 3D line.
- Allow mouse/touch rotation, pan, zoom, legend-based visibility toggling, and camera reset.
- Keep equal XYZ scaling so spatial shape is not distorted.
- Show file name, duration, row count, coordinate frame, and detected trajectories.
- Downsample only for rendering when a trajectory is very large; preserve first and last samples.
- Report missing columns, malformed rows, or files with no usable XYZ trajectory without crashing other uploads.

## Data Safety and Error Handling

- Never permanently delete rejected recordings.
- Never overwrite a file already present in quarantine; use a deterministic numeric suffix on collision.
- Exclude the quarantine directory and generated manifests from future scans.
- Fail clearly on a missing root or malformed `t` column; one bad episode does not prevent reporting the rest.

## Verification

- Unit tests cover duration calculation, strict threshold behavior (`<5`, not `<=5`), quarantine path construction, collision handling, and manifest output.
- Run the cleaner in dry-run mode, verify exactly two files are selected, then apply it and verify all remaining active episodes are at least 5 seconds.
- Open the viewer locally, load representative and malformed CSV fixtures, and verify trajectory rendering plus error states.

