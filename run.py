#!/usr/bin/env python3
"""Discover and run the retained dataset-specific research stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'causal'))
import project_paths

DATASETS = ('suning', 'amazon_appliances', 'amazon_beauty')
STAGES = {
    'dataset': ('MCRE/dataset_builder.py', 'MCRE/dataset_builder_amazon_{tag}.py'),
    'nuisance': ('DML/dcml_step1_nuisance_models.py', 'DML/dml_step1_nuisance_amazon_{tag}.py'),
    'inference': ('DML/dcml_step2_causal_inference.py', 'DML/dml_step2_causal_inference_amazon_{tag}.py'),
    'joint-tests': ('DML/rq2_joint_effect_tests.py', 'DML/rq2_joint_effect_tests_amazon_{tag}.py'),
    'moderation': ('DML/h3_moderation_interaction.py', 'DML/h3_moderation_interaction_amazon_{tag}.py'),
    'channels': ('DML/dcml_step3_channel_reporting.py', 'DML/dcml_step3_channel_reporting_amazon_{tag}.py'),
    'placebo': ('DML/placebo_test.py', 'DML/placebo_test_amazon_{tag}.py'),
    'causal-baselines': ('DML/causal_baseline_comparison.py', 'DML/causal_baseline_comparison_amazon_{tag}.py'),
    'specification': ('DML/specification_sensitivity.py', 'DML/specification_sensitivity_amazon_{tag}.py'),
    'gs-sensitivity': ('DML/gs_order_sensitivity.py', 'DML/gs_order_sensitivity_amazon_{tag}.py'),
    'ovb': ('DML/omitted_confounding_sensitivity.py', 'DML/omitted_confounding_sensitivity_amazon_{tag}.py'),
}


def stage_script(stage: str, dataset: str) -> Path:
    if dataset == 'suning':
        return ROOT / 'causal/model/M_suning' / STAGES[stage][0]
    tag = 'app' if dataset == 'amazon_appliances' else 'beauty'
    return ROOT / 'causal/model/M_amazon' / STAGES[stage][1].format(tag=tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='List available stages without running them')
    action.add_argument('--paths', action='store_true', help='Print the effective filesystem configuration')
    action.add_argument('--stage', choices=tuple(STAGES))
    parser.add_argument('--dataset', choices=DATASETS, default='suning')
    parser.add_argument('--dry-run', action='store_true', help='Show the command without executing it')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments after -- go to the research script')
    args = parser.parse_args()
    if args.paths:
        print(json.dumps(project_paths.describe(), indent=2))
        return
    if args.list:
        for stage in STAGES:
            path = stage_script(stage, args.dataset)
            print(f'{stage:18} {path.relative_to(ROOT)}')
        return
    script = stage_script(args.stage, args.dataset)
    if not script.is_file():
        raise FileNotFoundError(script)
    extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = [sys.executable, str(script), *extra]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
