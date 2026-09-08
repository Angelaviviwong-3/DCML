#!/usr/bin/env python3
"""Benchmark omitted-confounding sensitivity for final DCML DML estimates.

The formal nuisance models are first fitted with the full observed-covariate set.
For each benchmark group, the entire A-to-B and B-to-C nuisance sequence is then
refitted after omitting that observed group. Long-versus-short sensitivity
elements follow Chernozhukov et al.'s causal-ML framework. Reported one-, two-,
and three-benchmark bounds use adversarial rho=1 and user-cluster wild-bootstrap
uncertainty. They are sensitivity bounds, not randomized causal estimates.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor


VERSION = "20260728"
NUISANCE_SEED = 20260717
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcml_inference import (  # noqa: E402
    fit_apply_gs,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_nuisance import (  # noqa: E402
    dataset_contract,
    load_analysis_schema,
    make_classifier,
    make_regressor,
    numeric_matrix,
    predict_binary,
)
from dcml_utils import ensure_dir, read_table  # noqa: E402
from temporal_certification import filter_certified_frame, load_certification  # noqa: E402


SENSITIVITY_REFERENCE = "https://doi.org/10.3386/w30302"


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def benchmark_groups(columns: list[str]) -> dict[str, list[str]]:
    lower = {column: column.lower() for column in columns}
    groups = {
        "price": [column for column in columns if "price" in lower[column] or "x_pri" in lower[column]],
        "position": [column for column in columns if "position" in lower[column] or "x_pos" in lower[column]],
        "user_history": [
            column
            for column in columns
            if "user_" in lower[column]
            and ("_pre" in lower[column] or "tenure" in lower[column] or "history" in lower[column])
        ],
    }
    return {name: values for name, values in groups.items() if values}


def clean_name(column: str) -> str:
    return (
        column.replace("res_", "")
        .replace("__gs_pure", "")
        .replace("_calibrated", "")
        .replace("_calib", "")
    )


def load_certified_splits(dataset: str) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    contract = dataset_contract(ROOT, dataset)
    schema = load_analysis_schema(contract, dataset)
    certified_items, _, _ = load_certification(ROOT, dataset)
    splits = {
        "A": filter_certified_frame(read_table(contract["a"]), certified_items).reset_index(drop=True),
        "B": filter_certified_frame(read_table(contract["b"]), certified_items).reset_index(drop=True),
        "C": filter_certified_frame(read_table(contract["c"]), certified_items).reset_index(drop=True),
    }
    if dataset == "suning":
        for frame in splits.values():
            no_history = pd.to_numeric(frame["H_level"], errors="coerce").fillna(0).eq(0)
            frame.loc[no_history, "T_int_sem"] = np.nan
            frame["T_int_sem_available"] = (~no_history).astype(int)
    return splits, schema, contract


def treatment_residuals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    treatments: list[str],
    covariates: list[str],
    seed: int,
) -> pd.DataFrame:
    keep = ["user_id", "item_id", "timestamp", "H_level", "M_norm"]
    keep += [column for column in ["T_int_sem_available"] if column in test.columns]
    residuals = test[keep].copy()
    residuals["cluster_user_id"] = test["user_id"].astype(str)
    residuals["cluster_item_id"] = test["item_id"].astype(str)
    x_train = numeric_matrix(train, covariates)
    x_test = numeric_matrix(test, covariates)
    for index, treatment in enumerate(treatments):
        train_t = pd.to_numeric(train[treatment], errors="coerce")
        test_t = pd.to_numeric(test[treatment], errors="coerce")
        observed_train = train_t.notna()
        observed_test = test_t.notna()
        if not observed_train.any():
            residuals[f"res_{treatment}"] = np.nan
            continue
        if train_t.loc[observed_train].nunique() <= 1:
            model = DummyRegressor(strategy="constant", constant=float(train_t.loc[observed_train].iloc[0]))
        else:
            model = make_regressor(seed + index)
        model.fit(x_train.loc[observed_train], train_t.loc[observed_train])
        prediction = np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, 1.0)
        prediction[~observed_test.to_numpy()] = np.nan
        residuals[f"res_{treatment}"] = test_t.to_numpy(dtype=float) - prediction
        del model
    gc.collect()
    return residuals


def add_outcome_residuals(
    residuals: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcomes: list[str],
    covariates: list[str],
    seed: int,
) -> None:
    x_train = numeric_matrix(train, covariates)
    x_test = numeric_matrix(test, covariates)
    for index, outcome in enumerate(outcomes):
        train_y = pd.to_numeric(train[outcome], errors="coerce").fillna(0).astype(int)
        if train_y.nunique() <= 1:
            model = DummyClassifier(strategy="constant", constant=int(train_y.iloc[0]))
        else:
            model = make_classifier(seed + 100 + index)
        model.fit(x_train, train_y)
        prediction = np.clip(predict_binary(model, x_test), 0.0, 1.0)
        raw = pd.to_numeric(test[outcome], errors="coerce").fillna(0).to_numpy(dtype=float)
        residuals[f"res_{outcome}"] = raw - prediction
        del model
    gc.collect()


def refit_short_design(
    dataset: str,
    splits: dict[str, pd.DataFrame],
    schema: dict,
    contract: dict,
    omitted: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    treatments = list(schema["all_treatments"])
    gs_x = [column for column in schema.get("gs_confounder_whitelist", schema["confounder_whitelist"]) if column not in omitted]
    final_x = [column for column in schema["confounder_whitelist"] if column not in omitted]
    if not gs_x or not final_x:
        raise RuntimeError(f"{dataset}: benchmark omission removed the complete covariate set")
    gs_residuals = treatment_residuals(splits["A"], splits["B"], treatments, gs_x, seed)
    final_residuals = treatment_residuals(splits["B"], splits["C"], treatments, final_x, seed)
    add_outcome_residuals(final_residuals, splits["B"], splits["C"], contract["outcomes"], final_x, seed)
    final, composite_columns, _, _ = fit_apply_gs(
        gs_residuals,
        final_residuals,
        list(schema["aggregate_treatments"]),
        "Aggregate",
        "linear",
    )
    final, component_columns, _, _ = fit_apply_gs(
        gs_residuals,
        final,
        list(schema["disaggregate_treatments"]),
        "Disaggregate",
        "linear",
    )
    return final, {"Aggregate": composite_columns, "Disaggregate": component_columns}


def cluster_wild_sensitivity_draws(
    design: np.ndarray,
    residual: np.ndarray,
    treatment_residuals: np.ndarray,
    user: np.ndarray,
    beta: np.ndarray,
    sigma2: float,
    nu2: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reps <= 0:
        return beta[None, :], np.asarray([sigma2]), nu2[None, :]
    bread = np.linalg.pinv(design.T @ design)
    user_code, unique = pd.factorize(user.astype(str), sort=False)
    cluster_scores = np.zeros((len(unique), design.shape[1]), dtype=float)
    np.add.at(cluster_scores, user_code, design * residual[:, None])
    cluster_influence = cluster_scores @ bread
    del cluster_scores

    n = len(design)
    cluster_sigma = np.zeros(len(unique), dtype=float)
    np.add.at(cluster_sigma, user_code, (np.square(residual) - sigma2) / n)

    treatment_variance = np.divide(1.0, nu2)
    nu_influence = -np.divide(
        np.square(treatment_residuals) - treatment_variance[None, :],
        np.square(treatment_variance)[None, :] * n,
    )
    cluster_nu = np.zeros((len(unique), len(nu2)), dtype=float)
    np.add.at(cluster_nu, user_code, nu_influence)
    del nu_influence

    rng = np.random.default_rng(seed)
    beta_draws = np.empty((reps, design.shape[1]), dtype=float)
    sigma2_draws = np.empty(reps, dtype=float)
    nu2_draws = np.empty((reps, len(nu2)), dtype=float)
    for start in range(0, reps, 4):
        stop = min(reps, start + 4)
        weights = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, len(unique)))
        beta_draws[start:stop] = beta + weights @ cluster_influence
        sigma2_draws[start:stop] = sigma2 + weights @ cluster_sigma
        nu2_draws[start:stop] = nu2 + weights @ cluster_nu
    return (
        beta_draws,
        np.clip(sigma2_draws, 1e-12, None),
        np.clip(nu2_draws, 1e-12, None),
    )


def joint_sensitivity_elements(
    frame: pd.DataFrame,
    outcome: str,
    treatments: list[str],
    bootstrap_reps: int = 0,
    seed: int = 0,
) -> dict[str, dict]:
    complete = numeric(frame, [outcome, *treatments]).dropna()
    if len(complete) <= len(treatments) + 5:
        raise RuntimeError(f"Insufficient complete cases for {outcome}")
    y = complete[outcome].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(complete)), complete[treatments].to_numpy(dtype=float)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    sigma2 = float(np.mean(np.square(residual)))
    user = frame.loc[complete.index, "cluster_user_id"].astype(str).to_numpy()

    treatment_residuals = []
    nu2_values = []
    for treatment_index, treatment in enumerate(treatments):
        coefficient_index = treatment_index + 1
        d = design[:, coefficient_index]
        other_index = [index for index in range(design.shape[1]) if index != coefficient_index]
        other = design[:, other_index]
        d_residual = d - other @ np.linalg.lstsq(other, d, rcond=None)[0]
        d_variance = float(np.mean(np.square(d_residual)))
        if d_variance <= 0:
            raise RuntimeError(f"Degenerate Riesz representer for {treatment}")
        treatment_residuals.append(d_residual)
        nu2_values.append(1.0 / d_variance)

    nu2_array = np.asarray(nu2_values, dtype=float)
    beta_draws, sigma2_draws, nu2_draws = cluster_wild_sensitivity_draws(
        design,
        residual,
        np.column_stack(treatment_residuals),
        user,
        beta,
        sigma2,
        nu2_array,
        bootstrap_reps,
        seed,
    )

    output: dict[str, dict] = {}
    for treatment_index, treatment in enumerate(treatments):
        coefficient_index = treatment_index + 1
        coefficient_draws = beta_draws[:, coefficient_index]
        output[treatment] = {
            "N": int(len(complete)),
            "theta": float(beta[coefficient_index]),
            "sigma2": sigma2,
            "nu2": float(nu2_array[treatment_index]),
            "draws": coefficient_draws,
            "sigma2_draws": sigma2_draws,
            "nu2_draws": nu2_draws[:, treatment_index],
            "ci": np.percentile(coefficient_draws, [2.5, 97.5]),
        }
    return output


def gain_statistics(long: dict, short: dict) -> dict:
    sigma_long = long["sigma2"]
    sigma_short = short["sigma2"]
    nu_long = long["nu2"]
    nu_short = short["nu2"]
    cf_y = float(np.clip((sigma_short - sigma_long) / sigma_long, 0.0, 1.0))
    riesz_ratio = nu_short / nu_long
    cf_d = float(np.clip((1.0 - riesz_ratio) / riesz_ratio, 0.0, 1.0)) if riesz_ratio > 0 else 1.0
    delta_theta = float(short["theta"] - long["theta"])
    var_g = max(0.0, sigma_short - sigma_long)
    var_riesz = max(0.0, nu_long - nu_short)
    denominator = float(np.sqrt(var_g * var_riesz))
    rho = float(np.clip(abs(delta_theta) / denominator, 0.0, 1.0)) if denominator > 0 else 1.0
    rho *= float(np.sign(delta_theta) if delta_theta != 0 else 1.0)
    return {"cf_y": cf_y, "cf_d": cf_d, "rho": rho, "delta_theta": delta_theta}


def sensitivity_strength(cf_y: float, cf_d: float) -> float:
    """Return C_Y*C_D from DoubleML's cf_y and cf_d parameterization."""
    safe_cf_y = float(np.clip(cf_y, 0.0, 1.0))
    safe_cf_d = float(np.clip(cf_d, 0.0, 1.0 - 1e-9))
    return float(np.sqrt(safe_cf_y * safe_cf_d / (1.0 - safe_cf_d)))


def equal_strength_cf(distance: float, max_bias_scale: float) -> float:
    """Solve r/sqrt(1-r)=distance/(sigma*nu) for cf_y=cf_d=r."""
    if not np.isfinite(distance) or not np.isfinite(max_bias_scale) or max_bias_scale <= 0:
        return np.nan
    ratio = max(0.0, float(distance) / max_bias_scale)
    squared = ratio * ratio
    return float(np.clip((np.sqrt(squared * squared + 4.0 * squared) - squared) / 2.0, 0.0, 1.0))


def run(dataset: str, bootstrap_reps: int, seed: int) -> None:
    inference = inference_contract(ROOT, dataset)
    long_frame, long_specs, _, gs_manifest = prepare_orthogonalized_data(ROOT, dataset, "linear")
    splits, schema, nuisance_contract = load_certified_splits(dataset)
    groups = benchmark_groups(list(schema["confounder_whitelist"]))
    if not groups:
        raise RuntimeError(f"{dataset}: no price, position, or user-history benchmark group is available")

    long_models: dict[tuple[str, str], dict[str, dict]] = {}
    retained: dict[str, list[str]] = {}
    for specification, treatments in long_specs.items():
        retained[specification] = [column for column in treatments if support_diagnostics(long_frame, column)["Estimable"]]
        for outcome in inference["outcomes"]:
            long_models[(specification, outcome)] = joint_sensitivity_elements(
                long_frame,
                outcome,
                retained[specification],
                bootstrap_reps,
                seed + len(long_models) * 1000,
            )

    benchmark_rows: list[dict] = []
    benchmark_lookup: dict[tuple[str, str, str], list[dict]] = {}
    for group_name, omitted in groups.items():
        print(f"{dataset}: refitting nuisance models without {group_name}: {omitted}")
        short_frame, short_specs = refit_short_design(
            dataset,
            splits,
            schema,
            nuisance_contract,
            omitted,
            NUISANCE_SEED,
        )
        for specification in long_specs:
            if len(short_specs[specification]) != len(long_specs[specification]):
                raise RuntimeError(f"{dataset}: short and long {specification} specifications differ")
            for outcome in inference["outcomes"]:
                short_model = joint_sensitivity_elements(
                    short_frame,
                    outcome,
                    retained[specification],
                )
                stage = outcome.replace("res_y_", "")
                for treatment in retained[specification]:
                    long = long_models[(specification, outcome)][treatment]
                    short = short_model[treatment]
                    gain = gain_statistics(long, short)
                    row = {
                        "Dataset": dataset,
                        "Stage": stage,
                        "Specification": specification,
                        "Treatment": clean_name(treatment),
                        "Benchmark_Group": group_name,
                        "Omitted_Covariates": "|".join(omitted),
                        "Long_DML_ATE": long["theta"],
                        "Short_DML_ATE": short["theta"],
                        "Delta_ATE": gain["delta_theta"],
                        "cf_y": gain["cf_y"],
                        "cf_d": gain["cf_d"],
                        "rho_calibrated": gain["rho"],
                    }
                    benchmark_rows.append(row)
                    benchmark_lookup.setdefault((specification, outcome, treatment), []).append(row)
        del short_frame
        gc.collect()

    result_rows: list[dict] = []
    for (specification, outcome), model in long_models.items():
        stage = outcome.replace("res_y_", "")
        for treatment, long in model.items():
            candidates = benchmark_lookup[(specification, outcome, treatment)]
            strongest = max(candidates, key=lambda row: sensitivity_strength(row["cf_y"], row["cf_d"]))
            max_bias_scale = float(np.sqrt(long["sigma2"] * long["nu2"]))
            point_rv = equal_strength_cf(abs(long["theta"]), max_bias_scale)
            ci_lower, ci_upper = map(float, long["ci"])
            if ci_lower > 0:
                distance_to_zero = ci_lower
            elif ci_upper < 0:
                distance_to_zero = abs(ci_upper)
            else:
                distance_to_zero = 0.0
            significance_rv = equal_strength_cf(distance_to_zero, max_bias_scale)
            for multiplier in [0, 1, 2, 3]:
                cf_y = float(np.clip(multiplier * strongest["cf_y"], 0.0, 1.0))
                cf_d = float(np.clip(multiplier * strongest["cf_d"], 0.0, 1.0 - 1e-9))
                strength = sensitivity_strength(cf_y, cf_d)
                bias = max_bias_scale * strength
                direction = float(np.sign(long["theta"]) if long["theta"] != 0 else 1.0)
                adjusted = float(long["theta"] - direction * bias)
                bootstrap_bias = np.sqrt(long["sigma2_draws"] * long["nu2_draws"]) * strength
                shifted = long["draws"] - direction * bootstrap_bias
                adjusted_ci = np.percentile(shifted, [2.5, 97.5])
                result_rows.append(
                    {
                        "Dataset": dataset,
                        "Stage": stage,
                        "Specification": specification,
                        "Treatment": clean_name(treatment),
                        "N": long["N"],
                        "DML_ATE": long["theta"],
                        "Cluster_Wild_Bootstrap_CI_Lower": ci_lower,
                        "Cluster_Wild_Bootstrap_CI_Upper": ci_upper,
                        "Strongest_Observed_Benchmark": strongest["Benchmark_Group"],
                        "Benchmark_Multiplier": multiplier,
                        "Assumed_cf_y": cf_y,
                        "Assumed_cf_d": cf_d,
                        "Adversity_rho": 1.0,
                        "Worst_Case_Absolute_Bias": bias,
                        "Bias_Adjusted_ATE_Toward_Zero": adjusted,
                        "Bias_Adjusted_CI_Lower": float(adjusted_ci[0]),
                        "Bias_Adjusted_CI_Upper": float(adjusted_ci[1]),
                        "Equal_Strength_CF_To_Move_Point_Estimate_To_Zero": point_rv,
                        "Approx_Equal_Strength_CF_To_Move_Unadjusted_95CI_To_Zero": significance_rv,
                        "Interpretation": "causal-ML omitted-confounding bound; not a randomized estimate",
                    }
                )

    output = ensure_dir(_dcml_paths.workspace_path(ROOT, 'processed_data') / dataset / "Final_Causal_Output")
    result_path = output / f"{dataset}_DML_OVB_Sensitivity_{VERSION}.csv"
    benchmark_path = output / f"{dataset}_DML_OVB_Benchmark_Details_{VERSION}.csv"
    manifest_path = output / f"{dataset}_DML_OVB_Sensitivity_Manifest_{VERSION}.json"
    pd.DataFrame(result_rows).to_csv(result_path, index=False)
    pd.DataFrame(benchmark_rows).to_csv(benchmark_path, index=False)
    manifest = {
        "dataset": dataset,
        "version": VERSION,
        "method": "causal-ML omitted-variable-bias bounds using long-versus-short nuisance refits",
        "method_reference": SENSITIVITY_REFERENCE,
        "benchmark_groups": groups,
        "benchmark_multipliers": [1, 2, 3],
        "adversity_parameter": "rho=1 for reported worst-case bounds; calibrated rho retained in benchmark details",
        "nuisance_random_state": NUISANCE_SEED,
        "uncertainty": f"user-cluster wild bootstrap with {bootstrap_reps} replications",
        "gs_manifest": gs_manifest,
        "result_file": str(result_path),
        "benchmark_file": str(benchmark_path),
        "claim_boundary": "sensitivity analysis; it does not establish absence of unmeasured confounding",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {benchmark_path}")
    print(f"wrote {manifest_path}")


def main(default_dataset: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    if default_dataset is None:
        parser.add_argument("--dataset", choices=["suning", "amazon_appliances", "amazon_beauty"], required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    run(default_dataset or args.dataset, args.bootstrap_reps, args.seed)


if __name__ == "__main__":
    main()
