# Research run order

Commands below are run from the repository root with the relevant dependencies
installed. Data schemas and versioned input names are in [DATA.md](DATA.md).
Steps use the original research defaults. Outputs are local and ignored by Git.

## 1. Configure and prepare inputs

```bash
python tools/prepare_workspace.py
python run.py --paths
python tools/check_environment.py --group cpu
```

With the two Amazon datasets available:

```bash
python causal/data_preprocessing/amazon_appliances_preprocessing/process_confounders.py
python causal/data_preprocessing/amazon_appliances_preprocessing/process_item_reviews_amazon.py
python causal/data_preprocessing/amazon_appliances_preprocessing/build_y_funnel_amazon.py
python causal/data_preprocessing/amazon_appliances_preprocessing/download_images_amazon_app.py
python causal/data_preprocessing/amazon_beauty_preprocessing/download_images_amazon_beauty.py
```

With the private Suning workbooks available:

```bash
python causal/data_preprocessing/suning_preprocessing/process_confounder_user.py
python causal/data_preprocessing/suning_preprocessing/process_item_text_and_confounder.py
python causal/data_preprocessing/suning_preprocessing/process_item_reviews.py
python causal/data_preprocessing/suning_preprocessing/build_y_funnel.py
python causal/data_preprocessing/suning_preprocessing/download_images_suning.py
```

## 2. Multimodal measurement and split assembly

These extraction commands load Qwen2-VL and require model weights and a suitable
GPU environment. See each Amazon command's `--help` for slicing/resume controls.
Original sampled-generation defaults are retained.

```bash
python causal/model/M_suning/MCRE/extract_lmm_multimodal_feature_attention.py
python causal/model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon.py --dataset amazon_appliances
python causal/model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon.py --dataset amazon_beauty
python causal/model/M_amazon/MCRE/merge_multimodal_scalars_amazon.py
python run.py --stage dataset --dataset suning
python run.py --stage dataset --dataset amazon_appliances
python run.py --stage dataset --dataset amazon_beauty
```

Review the split diagnostics and source provenance. An available, matching frozen
measurement/split package may be used instead of re-extracting content; it must
retain the required filenames and schemas.

## 3. Temporal certification

```bash
python causal/model/audit_mcre_temporal_snapshot.py --dataset all
```

This produces per-item temporal eligibility and a summary. The full historical
sample may contain violations: the final analysis restricts **all three splits**
to eligible items through `temporal_certification.py`. Do not manufacture a pass
record or delete violating rows from an audit report. `--fail-on-violation` is
available when a fully clean input pool is required by a new study.

## 4. DML and formal analysis

Run the following sequence for each available dataset. This example uses Suning:

```bash
python run.py --stage nuisance --dataset suning
python run.py --stage inference --dataset suning
python run.py --stage joint-tests --dataset suning
python run.py --stage moderation --dataset suning
python run.py --stage channels --dataset suning
python run.py --stage placebo --dataset suning
python run.py --stage causal-baselines --dataset suning
python run.py --stage specification --dataset suning
python run.py --stage gs-sensitivity --dataset suning
python run.py --stage ovb --dataset suning -- --bootstrap-reps 1000
```

Replace `suning` with `amazon_appliances` or `amazon_beauty`. Individual scripts
with argument parsers accept their own flags after `--`; not every stage has a
parser. `--dry-run` on the launcher shows its command without running it.

Additional supporting routines are `nuisance_temporal_transport_diagnostics.py`,
`temporal_selection_sensitivity.py`, and Suning's `h4_stage_contrast.py`.
Older `dcml_step3_mediation*.py` scripts are retained as proxy-channel diagnostics;
the `channels` stage performs the later reporting workflow. They do not establish
an observed psychological mechanism.

After the dataset-level analyses, run the final robustness reporting:

```bash
python causal/postprocess_dml_ovb_sensitivity.py
python causal/summarize_rq5_robustness.py
python causal/Unified_Visualization/plot_dml_ovb_sensitivity.py
python causal/validate_rq5_outputs.py --fail-on-error
python causal/build_claim_aligned_robustness.py
python causal/Unified_Visualization/plot_claim_aligned_robustness.py
python causal/validate_claim_aligned_robustness.py --fail-on-error
```

Read failed, scope-limited, and sensitivity-limited checks as evidence boundaries.
These reports may require all three datasets; see their input paths before use.

## 5. Human measurement validation

```bash
python causal/Unified_Visualization/prepare_and_evaluate_mcre_human_validation.py --mode prepare
```

The prepare step writes blinded rater templates and a separate key under
`outputs/figures`. After three raters complete the templates:

```bash
python causal/Unified_Visualization/prepare_and_evaluate_mcre_human_validation.py --mode combine --rater-files outputs/figures/MCRE_Human_Annotation_Rater_A_20260728.csv outputs/figures/MCRE_Human_Annotation_Rater_B_20260728.csv outputs/figures/MCRE_Human_Annotation_Rater_C_20260728.csv
python causal/Unified_Visualization/prepare_and_evaluate_mcre_human_validation.py --mode evaluate --annotation-file outputs/figures/MCRE_Human_Annotations_20260728.csv --key-file outputs/figures/MCRE_Human_Annotation_Blinded_Key_20260728.csv
```

For custom output paths, replace these arguments with the paths printed by the
prepare step. The source's `--synthetic-pipeline-test` option is strictly a software
check; its `SYNTHETIC_DO_NOT_REPORT` outputs are not human evidence.

## 6. Recommendation evaluation

Inspect the retained script inventory before training:

```bash
python causal/baseline/A_base/run_suning_baselines.py --check-only
python causal/baseline/amazon/run_prediction_baselines_amazon.py --dataset amazon_appliances --check-only
```

For the full published multiseed design after all model dependencies and causal
inputs are available:

```bash
python causal/baseline/suning/run_suning_multiseed.py
python causal/baseline/amazon/run_amazon_prediction_multiseed_app.py
python causal/baseline/amazon/run_amazon_prediction_multiseed_beauty.py
```

Use the runners' `--help` for their controls. They retain the common candidate sets,
item/user hash checks, model-specific training implementations, and seed offsets.
Candidate sensitivity and feature ablations remain in `baseline/suning` and
`baseline/amazon`. Metrics from this block are predictive, not intervention effects.

## 7. Final supplement and figures

```bash
python causal/enhance/run_taxonomy_admissibility_sensitivity.py --dataset all
python causal/enhance/run_efficiency_cost_benchmark.py --mode summarize-existing
python causal/enhance/run_cross_mllm_measurement_audit.py --mode prepare
```

The cross-MLLM prepare step writes a fixed audit inventory. To score it with an
independently served alternative vision-language model:

```bash
python causal/enhance/run_cross_mllm_inference.py --api-base http://127.0.0.1:8000/v1 --model-id YOUR_SERVED_MODEL --resume
python causal/enhance/run_cross_mllm_measurement_audit.py --mode evaluate --model-id YOUR_SERVED_MODEL --alternative-output outputs/enhance/Cross_MLLM_Audit_Alternative_Scores_20260801.csv
```

An authenticated endpoint can use `DCML_MLLM_API_KEY`. Preparation/evaluation may
also need the frozen human-validation key, context inventory, and reference scores;
their paths can be supplied using the scripts' explicit options.

`Unified_Visualization/plot_figure_*.py` contains the retained final and supporting
figure producers. Their descriptive filenames are more informative than manuscript
figure numbers, which changed between revisions. The no-data measurement-map
figure can be generated with `plot_figure_5_mcre_measurement_graph.py`; other figures
read their specific frozen analysis outputs. No precomputed research figure is
included in this code-only release.
