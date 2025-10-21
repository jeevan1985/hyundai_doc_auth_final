#!/usr/bin/env python3
"""
Codebase audit tool: docstrings, type hints, and production-readiness checks.

Scans all Python files under a root directory and reports:
- Missing module/class/function docstrings
- Missing type hints on parameters and return types
- Bare except handlers and except-pass patterns
- print() usage (flagged for non-CLI/library files)
- TODO/FIXME comments

Usage:
  python tools/code_audit.py --root . --format text
  python tools/code_audit.py --root . --format json > audit.json

Exit codes:
  0 on success, 1 on internal error
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", "dist", "build",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache", ".warp"
}

CLI_HINT_FILES = {
    # Heuristics: common CLIs in this repo; adjust as needed
    "doc_image_verifier.py",
    "find_sim_images.py",
    "tif_search.py",
    "photo_extractor.py",
    "tool_search_tif_files_with_key.py",
    "scheduler_service.py",
}

@dataclass
class FunctionIssue:
    name: str
    lineno: int
    has_docstring: bool
    missing_params: List[str]
    missing_return: bool
    is_method: bool

@dataclass
class ClassIssue:
    name: str
    lineno: int
    has_docstring: bool

@dataclass
class FileReport:
    file: str
    has_module_docstring: bool
    functions: List[FunctionIssue]
    classes: List[ClassIssue]
    bare_except_count: int
    except_pass_count: int
    print_call_count: int
    todo_fixme_count: int

@dataclass
class Summary:
    total_files: int
    files_missing_module_docstring: int
    total_functions: int
    functions_missing_docstring: int
    functions_missing_types: int
    total_classes: int
    classes_missing_docstring: int
    bare_except_total: int
    except_pass_total: int
    print_calls_total: int
    todo_fixme_total: int


def is_ignored(path: Path) -> bool:
    parts = set(p.name for p in path.parents)
    return any(part in IGNORE_DIRS for part in (*parts, path.name))


class ParentAnnotator(ast.NodeVisitor):
    """Annotate nodes with parent references for method detection."""
    def __init__(self) -> None:
        self.parent_map: Dict[ast.AST, ast.AST] = {}

    def visit(self, node: ast.AST) -> Any:
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        return super().visit(node)

    def get_parent(self, node: ast.AST) -> Optional[ast.AST]:
        return self.parent_map.get(node)


def analyze_file(py_path: Path) -> Optional[FileReport]:
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except Exception:
        # Skip unparsable files from third-party/vendor areas
        return None

    parenter = ParentAnnotator()
    parenter.visit(tree)

    has_module_doc = ast.get_docstring(tree) is not None

    classes: List[ClassIssue] = []
    functions: List[FunctionIssue] = []

    bare_except = 0
    except_pass = 0
    print_calls = 0

    # Count TODO/FIXME in raw text
    lower_text = text.lower()
    todo_fixme = lower_text.count("todo") + lower_text.count("fixme")

    for node in ast.walk(tree):
        # Classes
        if isinstance(node, ast.ClassDef):
            classes.append(
                ClassIssue(
                    name=node.name,
                    lineno=node.lineno,
                    has_docstring=ast.get_docstring(node) is not None,
                )
            )
        # Functions / Methods
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parenter.get_parent(node)
            is_method = isinstance(parent, ast.ClassDef)

            # Parameters
            missing_params: List[str] = []
            args = []
            if node.args.posonlyargs:
                args.extend(node.args.posonlyargs)
            if node.args.args:
                args.extend(node.args.args)
            if node.args.kwonlyargs:
                args.extend(node.args.kwonlyargs)
            # vararg / kwarg (e.g., *args, **kwargs)
            if node.args.vararg and node.args.vararg.annotation is None:
                missing_params.append("*" + node.args.vararg.arg)
            if node.args.kwarg and node.args.kwarg.annotation is None:
                missing_params.append("**" + node.args.kwarg.arg)

            for a in args:
                # Allow untyped 'self'/'cls' on methods
                if is_method and a.arg in {"self", "cls"}:
                    continue
                if a.annotation is None:
                    missing_params.append(a.arg)

            missing_return = node.returns is None

            functions.append(
                FunctionIssue(
                    name=node.name,
                    lineno=node.lineno,
                    has_docstring=ast.get_docstring(node) is not None,
                    missing_params=missing_params,
                    missing_return=missing_return,
                    is_method=is_method,
                )
            )

        # Except handlers
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                bare_except += 1
            # except-pass pattern
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                except_pass += 1

        # print() calls
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "print":
                print_calls += 1

    return FileReport(
        file=str(py_path),
        has_module_docstring=has_module_doc,
        functions=functions,
        classes=classes,
        bare_except_count=bare_except,
        except_pass_count=except_pass,
        print_call_count=print_calls,
        todo_fixme_count=todo_fixme,
    )


def walk_py_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        files.append(p)
    return files


def build_summary(reports: List[FileReport]) -> Summary:
    total_files = len(reports)
    files_missing_module_docstring = sum(0 if r.has_module_docstring else 1 for r in reports)

    total_functions = sum(len(r.functions) for r in reports)
    functions_missing_docstring = sum(
        1 for r in reports for fn in r.functions if not fn.has_docstring
    )
    functions_missing_types = sum(
        1
        for r in reports
        for fn in r.functions
        if (fn.missing_params or fn.missing_return)
    )

    total_classes = sum(len(r.classes) for r in reports)
    classes_missing_docstring = sum(1 for r in reports for c in r.classes if not c.has_docstring)

    bare_except_total = sum(r.bare_except_count for r in reports)
    except_pass_total = sum(r.except_pass_count for r in reports)
    print_calls_total = sum(r.print_call_count for r in reports)
    todo_fixme_total = sum(r.todo_fixme_count for r in reports)

    return Summary(
        total_files=total_files,
        files_missing_module_docstring=files_missing_module_docstring,
        total_functions=total_functions,
        functions_missing_docstring=functions_missing_docstring,
        functions_missing_types=functions_missing_types,
        total_classes=total_classes,
        classes_missing_docstring=classes_missing_docstring,
        bare_except_total=bare_except_total,
        except_pass_total=except_pass_total,
        print_calls_total=print_calls_total,
        todo_fixme_total=todo_fixme_total,
    )


def render_text(reports: List[FileReport]) -> str:
    lines: List[str] = []
    summary = build_summary(reports)
    lines.append("=== Code Audit Summary ===")
    lines.append(f"Files scanned: {summary.total_files}")
    lines.append(
        f"Files missing module docstrings: {summary.files_missing_module_docstring}"
    )
    lines.append(
        f"Functions: {summary.total_functions} | missing docstrings: {summary.functions_missing_docstring} | missing types: {summary.functions_missing_types}"
    )
    lines.append(
        f"Classes: {summary.total_classes} | missing docstrings: {summary.classes_missing_docstring}"
    )
    lines.append(
        f"Bare excepts: {summary.bare_except_total} | except-pass: {summary.except_pass_total} | print() calls: {summary.print_calls_total} | TODO/FIXME: {summary.todo_fixme_total}"
    )
    lines.append("")

    def short(path: str) -> str:
        p = Path(path)
        try:
            return str(p.relative_to(Path.cwd()))
        except Exception:
            return path

    for r in reports:
        file_header_emitted = False

        def emit_header() -> None:
            nonlocal file_header_emitted
            if not file_header_emitted:
                lines.append(f"-- {short(r.file)}")
                if not r.has_module_docstring:
                    lines.append("   - Missing module docstring")
                file_header_emitted = True

        # Functions
        for fn in r.functions:
            issues: List[str] = []
            if not fn.has_docstring:
                issues.append("missing docstring")
            if fn.missing_params:
                issues.append(
                    "missing param types: " + ", ".join(fn.missing_params[:5]) + ("..." if len(fn.missing_params) > 5 else "")
                )
            if fn.missing_return:
                issues.append("missing return type")
            if issues:
                emit_header()
                kind = "method" if fn.is_method else "function"
                lines.append(f"   - {kind} {fn.name} (L{fn.lineno}): " + "; ".join(issues))

        # Classes
        for c in r.classes:
            if not c.has_docstring:
                emit_header()
                lines.append(f"   - class {c.name} (L{c.lineno}): missing docstring")

        # Other smells
        smells: List[str] = []
        if r.bare_except_count:
            smells.append(f"bare except x{r.bare_except_count}")
        if r.except_pass_count:
            smells.append(f"except-pass x{r.except_pass_count}")
        if r.print_call_count and Path(r.file).name not in CLI_HINT_FILES:
            smells.append(f"print() calls x{r.print_call_count}")
        if r.todo_fixme_count:
            smells.append(f"TODO/FIXME x{r.todo_fixme_count}")
        if smells:
            emit_header()
            lines.append("   - smells: " + ", ".join(smells))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args()

    root = args.root.resolve()
    py_files = walk_py_files(root)
    reports: List[FileReport] = []

    for f in py_files:
        if is_ignored(f):
            continue
        r = analyze_file(f)
        if r is not None:
            reports.append(r)

    # Sort reports by path
    reports.sort(key=lambda r: r.file)

    if args.format == "json":
        payload = {
            "summary": asdict(build_summary(reports)),
            "reports": [
                {
                    **{
                        k: v
                        for k, v in asdict(r).items()
                        if k not in {"functions", "classes"}
                    },
                    "functions": [asdict(f) for f in r.functions],
                    "classes": [asdict(c) for c in r.classes],
                }
                for r in reports
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(reports))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
