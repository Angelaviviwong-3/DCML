#!/usr/bin/env python3
"""Exercise the real inference helpers on artificial data, not study observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'causal'))

import numpy as np
import pandas as pd

from dcml_inference import fit_apply_gs, run_family

TREATMENTS = ['T_con_mkt', 'T_con_soc', 'T_con_rat', 'T_int_fac', 'T_int_vis', 'T_int_sem']


def synthetic_inputs(seed=42, n_train=1500, n_final=2400):
    rng = np.random.default_rng(seed)
    def sample(n):
        heuristic = rng.normal(size=(n, 3))
        independent = rng.normal(size=(n, 3))
        analytical = independent + heuristic @ np.array([[0.45, 0.2, 0.1], [0.1, 0.4, 0.2], [0.2, 0.1, 0.5]])
        frame = pd.DataFrame(np.column_stack([heuristic, analytical]), columns=['res_' + name for name in TREATMENTS])
        frame['res_y_purchase'] = 0.4 * heuristic[:, 0] + 0.7 * independent[:, 0] + rng.normal(scale=0.5, size=n)
        frame['cluster_user_id'] = ['synthetic_user_' + str(i % 160) for i in range(n)]
        frame['cluster_item_id'] = ['synthetic_item_' + str(x) for x in rng.integers(0, 120, n)]
        return frame
    return sample(n_train), sample(n_final)


def run_demo(seed=42):
    train, final = synthetic_inputs(seed)
    transformed, columns, diagnostics, manifest = fit_apply_gs(train, final, TREATMENTS, 'Aggregate', 'linear')
    rows = run_family(
        transformed, ['res_y_purchase'], columns, 'Aggregate', 'All (ATE)',
        bootstrap_reps=50, seed=seed, gs_mode='linear',
        claim_boundary='SYNTHETIC SOFTWARE DEMO; NOT EMPIRICAL EVIDENCE',
        temporal_analysis_population='artificial records; no certification claim',
    )
    result = pd.DataFrame(rows)
    result.insert(0, 'data_origin', 'SYNTHETIC_DEMO')
    target = result.loc[result['Component'].eq('T_int_fac'), 'Coefficient'].iloc[0]
    if not np.isfinite(target) or abs(target - 0.7) >= 0.08:
        raise RuntimeError(f'Synthetic effect recovery failed: expected near 0.7, found {target}')
    metadata = {
        'data_origin': 'SYNTHETIC_DEMO', 'seed': seed,
        'known_T_int_fac_effect': 0.7, 'estimated_T_int_fac_effect': float(target),
        'train_rows': len(train), 'final_rows': len(final), 'bootstrap_reps': 50,
        'note': 'Runs published inference helpers. Does not reproduce the paper or the full nuisance workflow.',
        'gs_training_manifest': manifest,
    }
    return result, pd.DataFrame(diagnostics), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args()
    results, diagnostics, metadata = run_demo()
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / 'synthetic_effects.csv', index=False)
    diagnostics.to_csv(args.output / 'synthetic_projection_diagnostics.csv', index=False)
    (args.output / 'synthetic_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('SYNTHETIC DEMO ONLY — not paper results')
    print(results[['Component', 'Coefficient', 'Estimable']].to_string(index=False))
    print(f'Known effect: 0.700; recovered effect: {metadata["estimated_T_int_fac_effect"]:.3f}')
    print(f'Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
