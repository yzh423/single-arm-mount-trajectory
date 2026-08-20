"""Stream only EgoDex HDF5 annotations from the remote test ZIP.

The Apple CDN supports HTTP byte ranges.  This downloader reads the ZIP central
directory, fetches each compressed HDF5 member directly (never MP4), converts
hand transforms to compact pose trajectories, and writes resumable shards plus
a single benchmark NPZ.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import h5py
import numpy as np

URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
ROOT = Path(__file__).resolve().parents[1]


class HTTPRange(io.RawIOBase):
    """Minimal seekable reader used only to obtain the ZIP central directory."""

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.size + offset
        else: raise ValueError(whence)
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.size: return b""
        if size < 0: size = self.size - self.pos
        end = min(self.size - 1, self.pos + size - 1)
        data = fetch_range(self.url, self.pos, end)
        self.pos += len(data)
        return data


def fetch_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise IOError(f"short range {len(data)} != {end-start+1}")
            return data
        except Exception:
            if attempt + 1 == retries: raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError


def member_bytes(info: zipfile.ZipInfo) -> bytes:
    # Local header: signature, version, flags, method, times, crc, sizes,
    # filename length, extra length.
    header = fetch_range(URL, info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50: raise zipfile.BadZipFile(info.filename)
    name_len, extra_len = fields[-2:]
    start = info.header_offset + 30 + name_len + extra_len
    compressed = fetch_range(URL, start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED: data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(compressed, -15)
    else: raise NotImplementedError(f"ZIP compression {info.compress_type}")
    if len(data) != info.file_size: raise IOError(f"size mismatch for {info.filename}")
    return data


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Vectorized, normalized rotation-matrix to wxyz quaternion conversion."""
    m = np.asarray(matrix, dtype=np.float64)
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    trace = np.trace(m, axis1=-2, axis2=-1)
    positive = trace > 0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-12)) * 2
    q[positive, 0] = .25 * s
    q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
    q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
    q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s
    remaining = ~positive
    for axis in range(3):
        other1, other2 = (axis + 1) % 3, (axis + 2) % 3
        mask = remaining & (m[..., axis, axis] >= m[..., other1, other1]) & (m[..., axis, axis] >= m[..., other2, other2])
        s = np.sqrt(np.maximum(1 + m[mask, axis, axis] - m[mask, other1, other1] - m[mask, other2, other2], 1e-12)) * 2
        q[mask, 0] = (m[mask, other2, other1] - m[mask, other1, other2]) / s
        q[mask, axis + 1] = .25 * s
        q[mask, other1 + 1] = (m[mask, other1, axis] + m[mask, axis, other1]) / s
        q[mask, other2 + 1] = (m[mask, other2, axis] + m[mask, axis, other2]) / s
        remaining &= ~mask
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    # Resolve q/-q frame discontinuities.
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0: q[i] *= -1
    return q.astype(np.float32)


def relative(reference: np.ndarray, poses: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(reference)
    return inverse @ poses


def pose_arrays(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:, :3, 3].astype(np.float32), matrix_to_quat_wxyz(transform[:, :3, :3])


def extract(info: zipfile.ZipInfo, destination: Path) -> dict:
    raw = member_bytes(info)
    with h5py.File(io.BytesIO(raw), "r") as handle:
        left = np.asarray(handle["transforms/leftHand"], dtype=np.float64)
        right = np.asarray(handle["transforms/rightHand"], dtype=np.float64)
        n = min(len(left), len(right))
        left, right = left[:n], right[:n]
        left_conf = np.asarray(handle["confidences/leftHand"][:n], dtype=np.float32) if "confidences/leftHand" in handle else np.ones(n, np.float32)
        right_conf = np.asarray(handle["confidences/rightHand"][:n], dtype=np.float32) if "confidences/rightHand" in handle else np.ones(n, np.float32)
        attrs = {str(k): str(v) for k, v in handle.attrs.items()}
    left_rel = relative(left[0], left)
    right_rel = relative(right[0], right)
    left_to_right = np.linalg.inv(left) @ right
    arrays = {"time_s": np.arange(n, dtype=np.float32) / 30.0,
              "left_confidence": left_conf, "right_confidence": right_conf}
    for prefix, transform in (("left_world", left), ("right_world", right),
                              ("left_relative", left_rel), ("right_relative", right_rel),
                              ("left_to_right", left_to_right)):
        arrays[prefix + "_xyz"], arrays[prefix + "_quat_wxyz"] = pose_arrays(transform)
    np.savez_compressed(destination, **arrays)
    return {"frames": n, "attributes": attrs, "source_uncompressed_bytes": len(raw)}


def consolidate(shard_dir: Path, episodes: list[dict], output: Path) -> None:
    keys = None
    values: dict[str, list[np.ndarray]] = {}
    offsets = [0]
    kept = []
    for episode in episodes:
        shard = shard_dir / episode["shard"]
        if not shard.exists(): continue
        with np.load(shard) as data:
            if keys is None:
                keys = list(data.files)
                values = {key: [] for key in keys}
            for key in keys: values[key].append(data[key])
            count = len(data["time_s"])
        offsets.append(offsets[-1] + count)
        kept.append(episode)
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    arrays["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["episode_task"] = np.asarray([x["task"] for x in kept])
    arrays["episode_source"] = np.asarray([x["source"] for x in kept])
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "EgoDex" / "pose_only_test")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(HTTPRange(URL)) as archive:
        members = [x for x in archive.infolist() if x.filename.lower().endswith((".hdf5", ".h5"))]
    if args.max_episodes: members = members[:args.max_episodes]
    episodes = [None] * len(members)
    manifest_path = args.output_dir / "manifest.json"
    pending = []
    for index, info in enumerate(members):
        relative_name = Path(info.filename)
        task = relative_name.parent.name
        shard_name = f"{index:05d}_{task}_{relative_name.stem}.npz"
        shard = shard_dir / shard_name
        metadata = {"index": index, "task": task, "source": info.filename, "shard": shard_name,
                    "compressed_bytes": info.compress_size}
        if shard.exists():
            with np.load(shard) as data: metadata["frames"] = len(data["time_s"])
            episodes[index] = metadata
        else:
            pending.append((index, info, shard, metadata))

    def process(item):
        index, info, shard, metadata = item
        metadata.update(extract(info, shard))
        return index, metadata

    completed = sum(x is not None for x in episodes)
    if completed:
        print(f"[EgoDex] resumed {completed}/{len(members)} existing shards", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            index, metadata = future.result()
            episodes[index] = metadata
            completed += 1
            if completed % 10 == 0 or completed == len(members):
                ready = [x for x in episodes if x is not None]
                manifest_path.write_text(json.dumps({"source_url": URL, "fps": 30, "episodes": ready}, indent=2), encoding="utf-8")
                downloaded = sum(x["compressed_bytes"] for x in ready) / 1e9
                print(f"[EgoDex] {completed}/{len(members)} episodes, {downloaded:.3f} GB HDF5 payload", flush=True)
    output = args.output_dir / "egodex_pose_only_test.npz"
    consolidate(shard_dir, episodes, output)
    summary = {"source_url": URL, "fps": 30, "episodes": len(episodes),
               "frames": int(sum(x["frames"] for x in episodes)), "output": output.name,
               "output_bytes": output.stat().st_size,
               "coordinate_note": "world transforms are in the stationary per-episode ARKit origin frame"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not args.keep_shards:
        # Shards intentionally remain by default during development; deletion is
        # left explicit to avoid losing resumable work after a failed merge.
        print(f"Resumable shards retained at {shard_dir}", flush=True)


if __name__ == "__main__":
    main()
