"""Release portability and a small numerical regression check."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def path_query(environment=None, cwd=None, expression='p.describe()'):
    env = {k: v for k, v in os.environ.items() if not k.startswith('DCML_')}
    env.update(environment or {})
    command = [sys.executable, '-c', f'import sys,json;sys.path.insert(0,{str(ROOT / "causal")!r});import project_paths as p;print(json.dumps({expression}))']
    result = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


class PortabilityTests(unittest.TestCase):
    def test_default_paths_do_not_follow_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = path_query(cwd=tmp)
        self.assertEqual(result['processed_data'], str(ROOT / 'data/processed'))
        self.assertEqual(result['Unified_Visualization/output'], str(ROOT / 'outputs/figures'))

    def test_configuration_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'paths.json'
            config.write_text(json.dumps({'processed_data': 'external-data/processed', 'log': str(Path(tmp) / 'logs')}))
            result = path_query({'DCML_PATH_CONFIG': str(config)}, cwd=tmp)
            self.assertEqual(result['processed_data'], str(ROOT / 'external-data/processed'))
            override = Path(tmp) / 'override'
            result = path_query({'DCML_PATH_CONFIG': str(config), 'DCML_PROCESSED_DATA_DIR': str(override)}, cwd=tmp)
            self.assertEqual(result['processed_data'], str(override.resolve()))

    def test_legacy_root_and_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = path_query({'DCML_CAUSAL_ROOT': tmp})
            self.assertEqual(result['processed_data'], str(Path(tmp).resolve() / 'processed_data'))
            explicit = path_query(expression=f'str(p.workspace_path({tmp!r},"processed_data","suning"))')
            self.assertEqual(explicit, str(Path(tmp).resolve() / 'processed_data/suning'))

    def test_inventory_and_nested_joins(self):
        result = path_query(expression='[str(p.workspace_path(p.CODE_ROOT,"processed_data/suning/Y/y_behavior_20260712.parquet")),str(p.workspace_path(p.REPOSITORY_ROOT,"causal","Unified_Visualization","output")),str(p.workspace_path(p.REPOSITORY_ROOT,"20260728_results","key.csv"))]')
        self.assertEqual(result, [str(ROOT / 'data/processed/suning/Y/y_behavior_20260712.parquet'), str(ROOT / 'outputs/figures'), str(ROOT / 'archives/20260728_results/key.csv')])

    def test_bad_config_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'paths.json'
            config.write_text('{"processed_dtaa":"typo"}')
            with self.assertRaises(subprocess.CalledProcessError):
                path_query({'DCML_PATH_CONFIG': str(config)})

    def test_static_release_contract(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/check_release.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class NumericalTests(unittest.TestCase):
    def test_categories_accept_english_and_original_labels(self):
        import pandas as pd
        sys.path.insert(0, str(ROOT / 'causal'))
        from dcml_utils import extract_category

        items = pd.DataFrame({
            'item_id': ['english', 'original', 'missing'],
            'text_W': [
                'Brand: Example | Category: Home-Appliances | Product name: Example',
                '\u54c1\u724c\uff1aExample | \u7c7b\u76ee\uff1aHome-Appliances | Product name: Example',
                None,
            ],
        })
        categories = extract_category(items, out_col='category')['category'].tolist()
        self.assertEqual(categories, ['Home-Appliances', 'Home-Appliances', 'Unknown'])

    def test_synthetic_effect_and_bootstrap(self):
        spec = importlib.util.spec_from_file_location('synthetic_example', ROOT / 'examples/synthetic_demo.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result, diagnostics, metadata = module.run_demo()
        self.assertEqual(len(result), 6)
        self.assertTrue(result['Estimable'].all())
        self.assertTrue(result['Bootstrap_CI_Lower_95'].notna().all())
        self.assertTrue((result['Bootstrap_CI_Lower_95'] < result['Bootstrap_CI_Upper_95']).all())
        self.assertTrue(diagnostics['Status'].eq('fitted_on_A_to_B_residuals').all())
        self.assertAlmostEqual(metadata['estimated_T_int_fac_effect'], 0.7, delta=0.08)


if __name__ == '__main__':
    unittest.main()
