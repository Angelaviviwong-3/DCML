#!/usr/bin/env python3
"""Check imports for the selected runtime group without loading models or data."""
import argparse
import importlib

GROUPS = {
    'cpu': ['numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'lightgbm', 'pyarrow', 'matplotlib', 'seaborn', 'joblib', 'openpyxl', 'requests', 'tqdm'],
    'mllm': ['torch', 'transformers', 'accelerate', 'qwen_vl_utils', 'PIL'],
    'baselines': ['torch', 'torch_geometric', 'tensorflow'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, default='cpu')
    args = parser.parse_args()
    failed = []
    for name in GROUPS[args.group]:
        try:
            module = importlib.import_module(name)
            print(f'OK {name}: {getattr(module, "__version__", "imported")}')
        except Exception as exc:
            print(f'FAIL {name}: {exc}')
            failed.append(name)
    if failed:
        raise SystemExit('Missing or unusable dependencies: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
