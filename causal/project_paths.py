"""Shared filesystem configuration for the publication copy.

Relative configured paths are resolved against the repository root, never the
current working directory. No directory is created when this module is imported.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULTS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'results_of_comparison': 'outputs/prediction',
    'log': 'outputs/logs',
    'Unified_Visualization/output': 'outputs/figures',
    'enhance/output': 'outputs/enhance',
    'archives': 'archives',
}
ENVIRONMENT = {
    'raw_data': 'DCML_RAW_DATA_DIR',
    'processed_data': 'DCML_PROCESSED_DATA_DIR',
    'results_of_comparison': 'DCML_PREDICTION_DIR',
    'log': 'DCML_LOG_DIR',
    'Unified_Visualization/output': 'DCML_FIGURE_DIR',
    'enhance/output': 'DCML_ENHANCE_DIR',
    'archives': 'DCML_ARCHIVE_DIR',
}


def absolute(value: str | os.PathLike) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def load_config() -> dict:
    explicit = os.environ.get('DCML_PATH_CONFIG')
    path = absolute(explicit) if explicit else REPOSITORY_ROOT / 'configs/paths.local.json'
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(f'DCML_PATH_CONFIG does not exist: {path}')
        return {}
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Path configuration must be a JSON object')
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown path configuration keys: {sorted(unknown)}')
    if not all(isinstance(value, str) and value.strip() for value in config.values()):
        raise ValueError('Configured paths must be nonempty strings')
    return config


CONFIG = load_config()


def project_path(relative: str = '') -> Path:
    """Resolve an original research path into the configured publication layout."""
    relative = relative.replace('\\', '/').strip('/')
    for key in sorted(DEFAULTS, key=len, reverse=True):
        if relative == key or relative.startswith(key + '/'):
            suffix = relative[len(key):].lstrip('/')
            override = os.environ.get(ENVIRONMENT[key]) or CONFIG.get(key)
            legacy = os.environ.get('DCML_CAUSAL_ROOT')
            if override:
                base = absolute(override)
            elif legacy and absolute(legacy) != CODE_ROOT:
                base = absolute(legacy) / key
            else:
                base = absolute(DEFAULTS[key])
            return base / suffix if suffix else base
    return CODE_ROOT / relative if relative else CODE_ROOT


def workspace_path(anchor: str | os.PathLike, *parts: str | os.PathLike) -> Path:
    """Honor explicit --causal-root arguments while relocating default paths."""
    path = absolute(anchor)
    relative_path = Path(*parts)
    if relative_path.is_absolute():
        return relative_path
    relative = relative_path.as_posix()
    if path == REPOSITORY_ROOT and relative.split('/')[0].endswith('_results'):
        return project_path('archives/' + relative)
    if path == REPOSITORY_ROOT and relative.startswith('causal/'):
        return project_path(relative[len('causal/'):])
    if path in {CODE_ROOT, REPOSITORY_ROOT}:
        if any(relative == key or relative.startswith(key + '/') for key in DEFAULTS):
            return project_path(relative)
        return path / relative_path
    return path / relative


def describe() -> dict[str, str]:
    return {key: str(project_path(key)) for key in DEFAULTS}


def source_version(script: str | os.PathLike) -> str:
    """Read a clean-named script's retained revision from the release inventory."""
    relative = Path(script).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    records = json.loads((REPOSITORY_ROOT / 'docs/source_manifest.json').read_text())
    for record in records:
        if record['release'] == relative:
            match = re.search(r'_(20\d{6})(?:_\d+)?\.py$', record['source'])
            return match.group(1) if match else 'unversioned'
    raise ValueError(f'Script is absent from the source inventory: {relative}')
