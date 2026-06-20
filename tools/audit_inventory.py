"""Audit and classify files before restructuring the MO-LSRTDE workspace."""

from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DIR = ROOT / "inventory"
CSV_PATH = INVENTORY_DIR / "files.csv"
JSON_PATH = INVENTORY_DIR / "files.json"
SUMMARY_PATH = INVENTORY_DIR / "summary.md"
SUMMARY_MARKER = "\n---\n\n## Latest Inventory Audit\n"

LARGE_FILE_BYTES = 5 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
TEXT_SCAN_BYTES = 512 * 1024

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_EXTENSIONS = {
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyd",
    ".pyc",
    ".so",
    ".whl",
    ".xls",
    ".xlsx",
    ".zip",
}
PAPER_EXTENSIONS = {".doc", ".docx", ".pdf"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx"}
FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".eps", ".pdf"}
NOTEBOOK_EXTENSIONS = {".ipynb"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"}
BUILD_EXTENSIONS = {".egg-info", ".pyc", ".pyo", ".pyd", ".so", ".dll"}

TRACKABLE_CATEGORIES = {
    "package_source",
    "algorithm_core",
    "algorithm_extension",
    "benchmark_problem",
    "metric",
    "experiment_script",
    "config",
    "test",
    "example",
    "generated_result_compact",
    "figure",
    "table",
    "documentation",
    "tooling",
}
ARCHIVE_OR_IGNORE_CATEGORIES = {
    "generated_result_raw",
    "notebook",
    "paper_or_manuscript",
    "publisher_pdf_or_copyright_risk",
    "private_or_sensitive",
    "cache_or_temp",
    "build_artifact",
    "environment_file",
    "unknown",
}


@dataclass(frozen=True)
class InventoryRecord:
    relative_path: str
    size_bytes: int
    extension: str
    sha256: str
    detected_type: str
    category: str
    confidence: str
    proposed_action: str
    proposed_target_path: str
    risk_flags: str


def relpath(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_git_internal(path: Path) -> bool:
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] == ".git"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_probably_binary(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True
    if suffix in TEXT_EXTENSIONS:
        return False
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return True
    if b"\0" in sample:
        return True
    if not sample:
        return False
    textish = sum(1 for byte in sample if byte in b"\n\r\t\f\b" or 32 <= byte <= 126)
    return textish / len(sample) < 0.75


def has_absolute_path_reference(path: Path, binary: bool) -> bool:
    if binary:
        return False
    try:
        data = path.read_bytes()[:TEXT_SCAN_BYTES]
    except OSError:
        return False
    text = data.decode("utf-8", errors="ignore")
    patterns = (
        r"[A-Za-z]:\\",
        r"/home/",
        r"/Users/",
        r"/mnt/",
        r"C:/",
        r"C:\\",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def detect_type(path: Path, binary: bool) -> str:
    suffix = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    if suffix == ".py":
        return "python_source"
    if suffix in CONFIG_EXTENSIONS:
        return "configuration"
    if suffix in NOTEBOOK_EXTENSIONS:
        return "jupyter_notebook"
    if suffix in PAPER_EXTENSIONS:
        return "document_or_pdf"
    if suffix in TABLE_EXTENSIONS:
        return "tabular_data"
    if suffix in FIGURE_EXTENSIONS:
        return "image_or_figure"
    if binary:
        return mime or "binary"
    return mime or "text"


def first_path_part(path: Path) -> str:
    try:
        return path.relative_to(ROOT).parts[0]
    except IndexError:
        return ""


def classify(path: Path, size: int, binary: bool) -> tuple[str, str]:
    relative = relpath(path).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    root_part = first_path_part(path).lower()

    if root_part == "archive":
        if relative.startswith("archive/papers/"):
            return "paper_or_manuscript", "high"
        if relative.startswith("archive/legacy_unclassified/"):
            return "unknown", "high"
        if relative.startswith("archive/legacy_code/") and suffix == ".py":
            return "algorithm_extension", "medium"
        if relative.startswith(("archive/legacy_audit/", "archive/legacy_codex_output/")):
            return "cache_or_temp", "high"
        return "unknown", "medium"
    if root_part == "docs" and suffix in {".md", ".rst"}:
        return "documentation", "high"
    if name in {"readme.md", "reproducibility.md", "changelog.md", "license"}:
        return "documentation", "high"
    if name in {".gitignore", ".gitattributes"}:
        return "config", "high"
    if root_part == "tools" and suffix == ".py":
        return "tooling", "high"
    if root_part in {".venv", "venv", "env"}:
        return "environment_file", "high"
    if root_part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox"}:
        return "cache_or_temp", "high"
    if "__pycache__" in path.parts or suffix in {".pyc", ".pyo"}:
        return "cache_or_temp", "high"
    if root_part in {".idea", ".vscode"}:
        return "private_or_sensitive", "medium"
    if root_part in {"build", "dist"} or suffix in BUILD_EXTENSIONS or ".egg-info/" in relative:
        return "build_artifact", "high"
    if root_part in {"results", "bench_result", "re_results"}:
        return "generated_result_raw", "high"
    if root_part in {"_codex_output"}:
        return "cache_or_temp", "high"
    if root_part in {"figures"} and suffix in FIGURE_EXTENSIONS:
        return "figure", "medium"
    if root_part in {"tables"} and suffix in TABLE_EXTENSIONS:
        return "table", "medium"
    if root_part in {"configs"} or (suffix in CONFIG_EXTENSIONS and "experiment" in name):
        return "config", "medium"
    if root_part in {"tests"} or name.startswith("test_") and suffix == ".py":
        return "test", "high"
    if root_part in {"examples"}:
        return "example", "high"
    if root_part in {"src"} and suffix == ".py":
        return "package_source", "high"
    if root_part in {"experiments"} and suffix == ".py":
        return "experiment_script", "high"
    if suffix in NOTEBOOK_EXTENSIONS:
        return "notebook", "high"
    if suffix in PAPER_EXTENSIONS:
        if suffix == ".pdf" and any(token in name for token in ("published", "publisher", "elsevier", "springer", "ieee", "acm")):
            return "publisher_pdf_or_copyright_risk", "medium"
        return "paper_or_manuscript", "high"
    if suffix in FIGURE_EXTENSIONS and suffix != ".pdf":
        return "figure", "medium" if size <= LARGE_FILE_BYTES else "low"
    if suffix in TABLE_EXTENSIONS:
        if root_part in {"bench_result", "re_results", "results"}:
            return "generated_result_raw", "high"
        return "table", "medium"
    if suffix == ".py":
        if name in {"mo_lsrtde.py", "algorithm.py"}:
            return "algorithm_core", "high"
        if name in {"metrics.py"} or "metric" in name:
            return "metric", "high"
        if name in {"problems.py", "reproblem.py", "thinfilm_problem.py"} or "problem" in name:
            return "benchmark_problem", "high"
        if name in {"run_experiments.py", "run_experiment.py"} or "experiment" in name:
            return "experiment_script", "high"
        if "visual" in name or "plot" in name or "postprocess" in name:
            return "experiment_script", "medium"
        return "algorithm_extension", "low"
    if suffix in CONFIG_EXTENSIONS:
        return "config", "medium"
    if root_part == "audit":
        return "generated_result_compact", "low"
    if suffix in {".txt", ".md", ".rst"}:
        return "unknown", "low"
    return "unknown", "low"


def proposed_target(path: Path, category: str) -> tuple[str, str]:
    relative = relpath(path)
    name = path.name
    suffix = path.suffix.lower()

    if category == "algorithm_core":
        return "move_to_package", "src/molsrtde/algorithm.py"
    if category == "metric":
        return "move_to_package", f"src/molsrtde/metrics/{name}"
    if category == "benchmark_problem":
        return "move_to_package", f"src/molsrtde/problems/{name}"
    if category == "algorithm_extension":
        return "review_then_move", f"src/molsrtde/extensions/{name}"
    if category == "experiment_script":
        if name == "run_experiments.py":
            return "refactor_to_experiment_entry", "scripts/run_experiment.py"
        return "move_to_experiments", f"experiments/molsrtde_experiments/{name}"
    if category == "config":
        return "move_or_keep_config", f"configs/{name}"
    if category == "test":
        return "move_or_keep_test", f"tests/{name}"
    if category == "example":
        return "track_public_example", relative
    if category == "figure":
        if path.stat().st_size <= LARGE_FILE_BYTES:
            return "review_then_track_compact", f"figures/{name}"
        return "archive_or_regenerate", f"archive/figures/{name}"
    if category == "table":
        if path.stat().st_size <= LARGE_FILE_BYTES:
            return "review_then_track_compact", f"tables/{name}"
        return "archive_or_regenerate", f"archive/tables/{name}"
    if category == "documentation":
        return "track_documentation", relative
    if category == "tooling":
        return "track_tooling", relative
    if category == "generated_result_compact":
        return "review_then_track_or_archive", relative
    if category == "generated_result_raw":
        return "keep_ignored_result", f"results/{name}"
    if category == "notebook":
        return "strip_or_archive", f"archive/legacy_notebooks/{name}"
    if category in {"paper_or_manuscript", "publisher_pdf_or_copyright_risk"}:
        return "archive_private_or_papers", f"archive/papers/{name}"
    if category == "private_or_sensitive":
        return "keep_ignored_private", f"private/{relative}"
    if category == "environment_file":
        return "ignore_environment", relative
    if category in {"cache_or_temp", "build_artifact"}:
        return "ignore_generated", relative
    return "preserve_unknown_for_review", f"archive/legacy_unclassified/{relative}"


def risk_flags(path: Path, size: int, binary: bool, category: str) -> list[str]:
    flags: list[str] = []
    suffix = path.suffix.lower()
    name = path.name.lower()
    if size > LARGE_FILE_BYTES:
        flags.append("large_file_over_5mb")
    if binary:
        flags.append("binary_file")
    if category in {"private_or_sensitive", "paper_or_manuscript"} or suffix in {".doc", ".docx"}:
        flags.append("possible_private_material")
    if category == "publisher_pdf_or_copyright_risk" or suffix == ".pdf":
        flags.append("possible_copyright_risk")
    if category in {"generated_result_raw", "generated_result_compact", "figure", "table", "cache_or_temp", "build_artifact"}:
        flags.append("generated_artifact")
    if category == "unknown":
        flags.append("unknown_purpose")
    if "private" in name or "password" in name or "secret" in name or "token" in name:
        flags.append("possible_private_material")
    if has_absolute_path_reference(path, binary):
        flags.append("absolute_path_reference")
    return sorted(set(flags))


def iter_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if dirname != ".git"]
        for filename in filenames:
            path = current / filename
            if is_git_internal(path):
                continue
            files.append(path)
    return sorted(files, key=lambda item: relpath(item).lower())


def build_records() -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    for path in iter_files():
        try:
            size = path.stat().st_size
            binary = is_probably_binary(path)
            digest = sha256_file(path)
            category, confidence = classify(path, size, binary)
            action, target = proposed_target(path, category)
            flags = risk_flags(path, size, binary, category)
            records.append(
                InventoryRecord(
                    relative_path=relpath(path),
                    size_bytes=size,
                    extension=path.suffix.lower(),
                    sha256=digest,
                    detected_type=detect_type(path, binary),
                    category=category,
                    confidence=confidence,
                    proposed_action=action,
                    proposed_target_path=target,
                    risk_flags=";".join(flags),
                )
            )
        except OSError as exc:
            records.append(
                InventoryRecord(
                    relative_path=relpath(path),
                    size_bytes=0,
                    extension=path.suffix.lower(),
                    sha256="",
                    detected_type=f"unreadable: {exc.__class__.__name__}",
                    category="unknown",
                    confidence="low",
                    proposed_action="preserve_unknown_for_review",
                    proposed_target_path=f"archive/legacy_unclassified/{relpath(path)}",
                    risk_flags="unknown_purpose",
                )
            )
    return records


def write_csv(records: list[InventoryRecord]) -> None:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_json(records: list[InventoryRecord]) -> None:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "records": [asdict(record) for record in records],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rows_for(records: list[InventoryRecord], categories: set[str]) -> list[InventoryRecord]:
    return [record for record in records if record.category in categories]


def render_table(records: list[InventoryRecord], columns: tuple[str, ...], limit: int) -> list[str]:
    if not records:
        return ["None."]
    lines = []
    for record in records[:limit]:
        values = []
        for column in columns:
            value = getattr(record, column)
            if column == "size_bytes":
                value = str(value)
            values.append(value)
        lines.append("- " + " | ".join(values))
    if len(records) > limit:
        lines.append(f"- ... {len(records) - limit} more not shown")
    return lines


def render_summary(records: list[InventoryRecord]) -> str:
    total_files = len(records)
    total_bytes = sum(record.size_bytes for record in records)
    largest = sorted(records, key=lambda record: record.size_bytes, reverse=True)[:20]
    category_counts = Counter(record.category for record in records)
    tracked = rows_for(records, TRACKABLE_CATEGORIES)
    archived_or_ignored = rows_for(records, ARCHIVE_OR_IGNORE_CATEGORIES)
    unknown = [record for record in records if record.category == "unknown"]

    lines = [
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "### Totals",
        f"- Total files: {total_files}",
        f"- Total bytes: {total_bytes}",
        "",
        "### Largest 20 Files",
        *render_table(largest, ("relative_path", "size_bytes", "category", "risk_flags"), 20),
        "",
        "### Category Counts",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "### Files Proposed for Git Tracking or Reviewable Public Tracking",
            f"- Count: {len(tracked)}",
            *render_table(tracked, ("relative_path", "category", "proposed_action", "proposed_target_path"), 80),
            "",
            "### Files Proposed for Archive, External, Private, or Ignore",
            f"- Count: {len(archived_or_ignored)}",
            *render_table(
                archived_or_ignored,
                ("relative_path", "category", "proposed_action", "proposed_target_path", "risk_flags"),
                120,
            ),
            "",
            "### Unknown Files Requiring Human Review",
            f"- Count: {len(unknown)}",
            *render_table(unknown, ("relative_path", "proposed_target_path", "risk_flags"), 120),
            "",
            "### Ticket 01 Commands Run",
            "- `New-Item -ItemType Directory -Force -Path tools`",
            "- `python tools/audit_inventory.py`",
            "",
            "### Ticket 01 Remaining Risk",
            "- Classifications are conservative and based on names, paths, extensions, binary detection, and lightweight content scans.",
            "- Unknown files are preserved for human review and must not be discarded.",
            "- Publication-risk materials are proposed for archive/private handling, not public tracking.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary(records: list[InventoryRecord]) -> None:
    base = ""
    if SUMMARY_PATH.exists():
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        base = existing.split(SUMMARY_MARKER, 1)[0].rstrip() + "\n"
    SUMMARY_PATH.write_text(base + SUMMARY_MARKER + render_summary(records), encoding="utf-8")


def main() -> None:
    records = build_records()
    if not records:
        raise SystemExit("No files found to inventory.")
    write_csv(records)
    write_json(records)
    write_summary(records)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Total files: {len(records)}")
    print(f"Total bytes: {sum(record.size_bytes for record in records)}")


if __name__ == "__main__":
    main()
