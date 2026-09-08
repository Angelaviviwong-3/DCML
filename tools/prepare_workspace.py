#!/usr/bin/env python3
"""Create configured output/data folders and install the frozen Suning schema."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'causal'))
import project_paths


def main():
    for path in project_paths.describe().values():
        Path(path).mkdir(parents=True, exist_ok=True)
    target = project_paths.project_path('processed_data/suning/DML_Results/DML_feature_schema_20260712.json')
    if target.exists():
        print(f'Existing schema preserved: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'configs/suning_feature_schema.json', target)
        print(f'Installed frozen structural schema: {target}')
    print('Workspace prepared. No dataset observations were installed.')


if __name__ == '__main__':
    main()
