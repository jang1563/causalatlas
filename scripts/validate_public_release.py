#!/usr/bin/env python3
"""Validate the public CausalAtlas release surface.

The checks here are intentionally lightweight: they validate machine-readable
artifacts, ensure the manifest points to files that exist, and guard against
publishing excluded local/runtime materials.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]

STRICT_JSON_FILES = [
    "artifact_manifest.json",
    "schemas/artifact_manifest.schema.json",
    "schemas/layerA_norman.schema.json",
    "schemas/layerA_tahoe.schema.json",
    "results/move1/layerA_norman.json",
    "results/move1/layerA_tahoe.json",
]

_cat = "".join
FORBIDDEN_PATH_RE = re.compile(
    "|".join(
        [
            r"(^|/)_private/",
            r"(^|/)eval/p1_",
            r"(^|/)eval/p6_",
            r"(^|/)eval/harness/",
            r"\.slurm$",
            r"\.sbatch$",
            "REVIEW",
            _cat(["HAND", "OFF"]),
            _cat(["JD", "_"]),
            "PILOT",
            _cat(["PLAN", "_VS"]),
            _cat(["DESIGN", "_v1"]),
            "REANCHOR",
            r"PHASE[0-9]",
            "_metrics",
            "agent_eval",
            _cat(["PREPRINT", "_INTRO"]),
            _cat(["ARC", "_STATE"]),
            _cat(["ASSET", "_DEP"]),
            "CALIBRATED",
            _cat(["CAUSAL", "ATLAS_"]),
        ]
    ),
    re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".json",
    ".md",
    ".py",
    ".r",
    ".R",
    ".sh",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def tracked_files() -> list[str]:
    return [line for line in run_git(["ls-files"]).splitlines() if line]


def load_strict_json(path: pathlib.Path) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def local_manifest_paths(manifest: dict) -> set[str]:
    paths: set[str] = set()
    paths.update(manifest.get("entrypoints", {}).values())
    paths.update(item["path"] for item in manifest.get("result_artifacts", []))
    paths.update(item["path"] for item in manifest.get("schema_files", []))
    paths.update(item["path"] for item in manifest.get("pipeline_files", []))
    for claim in manifest.get("claims", []):
        for key in ("evidence_files", "code_files"):
            paths.update(p for p in claim.get(key, []) if not p.startswith("http"))
    return paths


def scan_terms() -> list[str]:
    join = "".join
    return [
        join(["RS/", "faculty"]),
        join(["faculty ", "application"]),
        join(["hir", "ing", "-lens"]),
        join(["AI_", "Grant"]),
        join(["the ", "JD"]),
        join(["JD ", "lens"]),
        join(["Anthropic ", "Life"]),
        join(["Debt of ", "Negative"]),
        join(["neg", "biodb"]),
        join(["anthropic_", "api_key"]),
        join(["grant ", "workspace"]),
        join(["job ", "application"]),
        join(["cover ", "letter"]),
        join(["preprint/", "application"]),
        join(["private ", "planning"]),
        join(["at your ", "request"]),
    ]


def iter_text_files(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    for rel in paths:
        path = ROOT / rel
        if path.suffix in TEXT_SUFFIXES:
            yield path


def check_strict_json() -> None:
    for rel in STRICT_JSON_FILES:
        path = ROOT / rel
        if not path.exists():
            fail(f"missing JSON artifact: {rel}")
        load_strict_json(path)
    print("strict JSON OK")


def check_manifest() -> None:
    manifest = load_strict_json(ROOT / "artifact_manifest.json")
    if not isinstance(manifest, dict):
        fail("artifact_manifest.json must contain an object")

    missing = sorted(p for p in local_manifest_paths(manifest) if not (ROOT / p).exists())
    if missing:
        fail("manifest points to missing files: " + ", ".join(missing))

    schema_by_result = {
        item["path"]: item.get("schema")
        for item in manifest.get("result_artifacts", [])
        if item.get("type") == "json"
    }
    for result_path, schema_path in schema_by_result.items():
        if not schema_path:
            fail(f"JSON result artifact lacks schema: {result_path}")
        if not (ROOT / schema_path).exists():
            fail(f"schema missing for {result_path}: {schema_path}")

    print("manifest OK")


def check_tracked_file_set(paths: list[str]) -> None:
    offenders = [p for p in paths if FORBIDDEN_PATH_RE.search(p)]
    if offenders:
        fail("forbidden tracked paths: " + ", ".join(offenders))
    print("tracked file set OK")


def check_text_content(paths: list[str]) -> None:
    terms = scan_terms()
    offenders: list[str] = []
    for path in iter_text_files(paths):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in terms:
            if term in text:
                offenders.append(f"{path.relative_to(ROOT)}: {term}")
    if offenders:
        fail("forbidden public text found: " + "; ".join(offenders))
    print("public text OK")


def main() -> None:
    paths = tracked_files()
    check_strict_json()
    check_manifest()
    check_tracked_file_set(paths)
    check_text_content(paths)


if __name__ == "__main__":
    main()
