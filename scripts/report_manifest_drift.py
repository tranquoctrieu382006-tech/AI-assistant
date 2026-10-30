"""Report frozen-manifest drift without failing fast.

Used before regenerating a baseline: shows which recorded SHA-256 values no
longer match the working tree under the portable (LF-normalised) hash format.
Exit code is 0 only when every entry matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.manifest_hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = (
    "eval/development_manifest_m1.json",
    "eval/heldout_manifest_m2.json",
)


def _entries(manifest: dict) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for key in ("corpus_files", "evaluation_code", "heldout_evaluator_code"):
        for relative, expected in manifest.get(key, {}).items():
            entries.append((relative, expected))
    for relative, info in manifest.get("development_sets", {}).items():
        entries.append((relative, info["sha256"]))
    if manifest.get("heldout_path"):
        entries.append((manifest["heldout_path"], manifest["heldout_sha256"]))
    return entries


def _line_endings(raw: bytes) -> str:
    crlf = raw.count(b"\r\n")
    lf_only = raw.count(b"\n") - crlf
    if crlf and lf_only:
        return f"mixed(crlf={crlf},lf={lf_only})"
    if crlf:
        return "crlf"
    if lf_only:
        return "lf"
    return "none"


def collect(manifest_path: Path, root: Path = ROOT) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for relative, expected in _entries(manifest):
        path = root / relative
        if not path.exists():
            rows.append({"path": relative, "status": "missing", "expected": expected})
            continue
        raw = path.read_bytes()
        canonical = sha256_file(path)
        raw_sha = hashlib.sha256(raw).hexdigest()
        rows.append(
            {
                "path": relative,
                "status": "ok" if canonical == expected else "drift",
                "expected": expected,
                "canonical_sha256": canonical,
                "raw_sha256": raw_sha,
                "raw_matches_expected": raw_sha == expected,
                "line_endings": _line_endings(raw),
            }
        )
    return rows


def main(paths: list[Path]) -> int:
    drifted = 0
    for manifest_path in paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = manifest.get("manifest", manifest_path.stem)
        print(f"== {manifest_path.relative_to(ROOT).as_posix()} ({label})")
        print(f"   hash_format: {manifest.get('hash_format', '<not recorded>')}")
        rows = collect(manifest_path)
        for row in rows:
            if row["status"] == "missing":
                print(f"   MISSING  {row['path']}")
                continue
            note = "ok" if row["status"] == "ok" else "DRIFT"
            legacy = "" if row["status"] == "ok" else (
                " (raw bytes still match)" if row["raw_matches_expected"] else ""
            )
            print(f"   {note:<7} {row['path']}  [{row['line_endings']}]{legacy}")
        manifest_drift = sum(1 for row in rows if row["status"] != "ok")
        drifted += manifest_drift
        print(f"   entries={len(rows)} drift={manifest_drift}\n")
    return 0 if drifted == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", dest="manifests", type=Path)
    args = parser.parse_args()
    selected = args.manifests or [ROOT / name for name in DEFAULT_MANIFESTS]
    raise SystemExit(main([p if p.is_absolute() else ROOT / p for p in selected]))
