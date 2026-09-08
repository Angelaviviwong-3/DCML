#!/usr/bin/env python3
"""Check the public code layout without datasets or third-party dependencies."""
from __future__ import annotations

import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def definitions(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
    return names


def check():
    problems = []
    files = sorted(p for folder in ('causal', 'examples', 'tools', 'tests') for p in (ROOT / folder).rglob('*.py')) + [ROOT / 'run.py']
    modules = defaultdict(list)
    trees = {}
    for path in files:
        if re.search(r'_20\d{6}', str(path.relative_to(ROOT))):
            problems.append(f'Dated code name: {path.relative_to(ROOT)}')
        source = path.read_text(encoding='utf-8')
        try:
            trees[path] = ast.parse(source, filename=str(path))
            compile(source, str(path), 'exec')
        except SyntaxError as exc:
            problems.append(str(exc))
        if path.is_relative_to(ROOT / 'causal'):
            modules[path.stem].append(path)
        # Personal workstation roots must not be part of the publication code.
        if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/', source):
            problems.append(f'Personal absolute path: {path.relative_to(ROOT)}')
    code_names = {path.name for path in files}
    imports_checked = 0
    scripts_checked = 0
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in modules:
                choices = modules[node.module]
                target = next((p for p in choices if p.parent == path.parent), choices[0])
                available = definitions(trees[target])
                for alias in node.names:
                    imports_checked += 1
                    if alias.name != '*' and alias.name not in available:
                        problems.append(f'{path.relative_to(ROOT)} imports missing {node.module}.{alias.name}')
            if isinstance(node, ast.ImportFrom) and node.module and re.search(r'_20\d{6}', node.module):
                problems.append(f'Dated import in {path.relative_to(ROOT)}: {node.module}')
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.endswith('.py') and not any(c.isspace() or c in '*{}' for c in value) and not value.startswith('.'):
                    scripts_checked += 1
                    if Path(value).name not in code_names:
                        problems.append(f'Missing script reference in {path.relative_to(ROOT)}: {value}')
    sys.path.insert(0, str(ROOT))
    import run
    for dataset in run.DATASETS:
        for stage in run.STAGES:
            if not run.stage_script(stage, dataset).is_file():
                problems.append(f'Missing launcher stage: {dataset}/{stage}')
    manifest = json.loads((ROOT / 'docs/source_manifest.json').read_text())
    for record in manifest:
        if not (ROOT / record['release']).is_file():
            problems.append(f'Missing manifest entry: {record["release"]}')
    return {'python_files': len(files), 'selected_source_scripts': len(manifest), 'local_import_symbols': imports_checked, 'script_literals': scripts_checked, 'launcher_entries': len(run.DATASETS) * len(run.STAGES), 'problems': problems}


if __name__ == '__main__':
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['problems'] else 0)
