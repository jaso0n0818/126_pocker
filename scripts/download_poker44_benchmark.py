#!/usr/bin/env python3
"""Download released Poker44 training benchmark data into a local register."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_API_BASE = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUTPUT_DIR = Path("download/poker44_benchmark")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_json(url: str, *, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_release(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") or {}
    release_rows = data.get("chunks") or []
    chunks: list[Any] = []
    ground_truth: list[int] = []
    ground_truth_labels: list[str] = []
    source_batch_ids: list[str] = []

    for row in release_rows:
        row_chunks = list(row.get("chunks") or [])
        row_truth = list(row.get("groundTruth") or [])
        row_labels = list(row.get("groundTruthLabels") or [])
        if len(row_chunks) != len(row_truth):
            raise ValueError(
                f"Batch {row.get('chunkId')} has {len(row_chunks)} chunks "
                f"but {len(row_truth)} labels"
            )
        chunks.extend(row_chunks)
        ground_truth.extend(int(label) for label in row_truth)
        ground_truth_labels.extend(str(label) for label in row_labels)
        source_batch_ids.extend([str(row.get("chunkId") or "")] * len(row_chunks))

    return {
        "source": "poker44_training_benchmark",
        "sourceDate": data.get("sourceDate"),
        "releaseVersion": data.get("releaseVersion"),
        "cutoffWindowStart": data.get("cutoffWindowStart"),
        "downloadedAt": _utc_now(),
        "chunks": chunks,
        "groundTruth": ground_truth,
        "groundTruthLabels": ground_truth_labels,
        "sourceBatchIds": source_batch_ids,
    }


def _dataset_stats(payload: dict[str, Any]) -> dict[str, Any]:
    labels = payload.get("groundTruth") or []
    chunks = payload.get("chunks") or []
    chunk_sizes = [len(chunk) for chunk in chunks if isinstance(chunk, list)]
    return {
        "chunk_count": len(chunks),
        "hand_count": sum(chunk_sizes),
        "human_chunks": sum(1 for label in labels if int(label) == 0),
        "bot_chunks": sum(1 for label in labels if int(label) == 1),
        "min_hands_per_chunk": min(chunk_sizes) if chunk_sizes else 0,
        "max_hands_per_chunk": max(chunk_sizes) if chunk_sizes else 0,
    }


def _write_register(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Poker44 released benchmark chunks into a local register."
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-date",
        action="append",
        help="Download only this sourceDate. Can be passed multiple times.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    status = _fetch_json(args.api_base)
    releases_payload = _fetch_json(f"{args.api_base}/releases")
    releases = list((releases_payload.get("data") or {}).get("releases") or [])
    wanted_dates = set(args.source_date or [])
    if wanted_dates:
        releases = [release for release in releases if release.get("sourceDate") in wanted_dates]
    if not releases:
        raise SystemExit("No matching Poker44 benchmark releases found.")

    register_entries: list[dict[str, Any]] = []
    combined_chunks: list[Any] = []
    combined_truth: list[int] = []
    combined_truth_labels: list[str] = []

    for release in releases:
        source_date = str(release["sourceDate"])
        target = output_dir / f"poker44_benchmark_{source_date}.json.gz"
        if target.exists():
            print(f"Using existing Poker44 benchmark release {source_date}...", flush=True)
            flattened = _read_gzip_json(target)
        else:
            print(f"Downloading Poker44 benchmark release {source_date}...", flush=True)
            release_payload = _fetch_json(f"{args.api_base}/chunks?sourceDate={source_date}")
            flattened = _flatten_release(release_payload)
            _write_gzip_json(target, flattened)

        stats = _dataset_stats(flattened)

        combined_chunks.extend(flattened["chunks"])
        combined_truth.extend(flattened["groundTruth"])
        combined_truth_labels.extend(flattened["groundTruthLabels"])

        register_entries.append(
            {
                "sourceDate": source_date,
                "releaseVersion": flattened.get("releaseVersion"),
                "path": str(target),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
                **stats,
            }
        )

    combined = {
        "source": "poker44_training_benchmark",
        "sourceDates": [entry["sourceDate"] for entry in register_entries],
        "releaseVersion": (status.get("data") or {}).get("releaseVersion"),
        "downloadedAt": _utc_now(),
        "chunks": combined_chunks,
        "groundTruth": combined_truth,
        "groundTruthLabels": combined_truth_labels,
    }
    combined_path = output_dir / "poker44_benchmark_all.json.gz"
    _write_gzip_json(combined_path, combined)

    register = {
        "apiBase": args.api_base,
        "createdAt": _utc_now(),
        "status": status.get("data") or {},
        "combined": {
            "path": str(combined_path),
            "sha256": _sha256_file(combined_path),
            "bytes": combined_path.stat().st_size,
            **_dataset_stats(combined),
        },
        "releases": register_entries,
    }
    register_path = output_dir / "register.json"
    _write_register(register_path, register)

    print(f"Wrote register: {register_path}")
    print(f"Wrote combined dataset: {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
