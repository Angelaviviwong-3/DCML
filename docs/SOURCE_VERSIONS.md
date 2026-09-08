# Source versions and publication changes

This release selects one latest script per filename family from the source `causal` tree.
Full date suffixes are compared numerically; an additional revision number resolves
same-day variants. Code-directory date suffixes are removed as well. The source
workspace is unchanged.

The release contains **194 retained research scripts**. Local import references,
literal and dynamically constructed script filenames, and code directory references
were updated to match the public names. Data/result basenames and experiment
version constants were preserved. Original servers' absolute paths were replaced
with `project_paths.py`; MLLM model paths and API keys accept environment overrides.
Baseline runners verify source revisions against `source_manifest.json`, since a
cleaned filename no longer contains its original date.

`source_manifest.json` records the exact original SHA-256 for every script below.
`excluded_sources.json` records omitted source scripts, including older versions,
exploratory diagnostics, and unused upstream checkouts. No private data or result
archive is included.

An additional structural input, `configs/suning_feature_schema.json`, preserves
the decoded schema from `202607_results_suning/20260712_results/DML_feature_schema_20260712.json`.
It contains field lists, not user/item observations. The original source file's
SHA-256 is `55755ddc5d1da8aa8a4648d34055e4097b32d0065fe6230bcad99043ce231f20`.
The English publication uses JSON Unicode escapes for original category identifiers,
so its byte-level checksum differs while the decoded schema remains unchanged.

The 2026-09-09 update uses the 233-file local DCML package, which matched its ZIP
archive exactly before publication edits. The Chinese README was removed; comments,
messages, help text, and the Suning extraction prompt were translated to English.
Category parsing supports both old and new text labels. See the
[reproducibility notes](REPRODUCIBILITY.md#english-publication-changes) for the effect
of prompt translation on newly extracted measurements.

## Retained scripts

| Public path | Original path within causal |
| --- | --- |
| `causal/Unified_Visualization/OPE_Policy.py` | `Unified_Visualization/OPE_Policy_20260713.py` |
| `causal/Unified_Visualization/plot_claim_aligned_robustness.py` | `Unified_Visualization/plot_claim_aligned_robustness_20260729_2.py` |
| `causal/Unified_Visualization/plot_dml_ovb_sensitivity.py` | `Unified_Visualization/plot_dml_ovb_sensitivity_20260729.py` |
| `causal/Unified_Visualization/plot_figure_10_causal_baseline_comparison.py` | `Unified_Visualization/plot_figure_10_causal_baseline_comparison_20260727.py` |
| `causal/Unified_Visualization/plot_figure_10_prediction_performance.py` | `Unified_Visualization/plot_figure_10_prediction_performance_20260721.py` |
| `causal/Unified_Visualization/plot_figure_11_ate_specification_sensitivity.py` | `Unified_Visualization/plot_figure_11_ate_specification_sensitivity_20260728.py` |
| `causal/Unified_Visualization/plot_figure_11_gs_sensitivity.py` | `Unified_Visualization/plot_figure_11_gs_sensitivity_20260721.py` |
| `causal/Unified_Visualization/plot_figure_5_mcre_measurement_graph.py` | `Unified_Visualization/plot_figure_5_mcre_measurement_graph_20260723.py` |
| `causal/Unified_Visualization/plot_figure_6_identification_diagnostics.py` | `Unified_Visualization/plot_figure_6_identification_diagnostics_20260721.py` |
| `causal/Unified_Visualization/plot_figure_6_rq1_cross_path_orthogonalization.py` | `Unified_Visualization/plot_figure_6_rq1_cross_path_orthogonalization_20260722.py` |
| `causal/Unified_Visualization/plot_figure_6_rq1_dual_specification_heatmap.py` | `Unified_Visualization/plot_figure_6_rq1_dual_specification_heatmap_20260722.py` |
| `causal/Unified_Visualization/plot_figure_6_rq1_semantic_support.py` | `Unified_Visualization/plot_figure_6_rq1_semantic_support_20260722.py` |
| `causal/Unified_Visualization/plot_figure_7_rq1_gs_orthogonalization.py` | `Unified_Visualization/plot_figure_7_rq1_gs_orthogonalization_20260722.py` |
| `causal/Unified_Visualization/plot_figure_7_rq2_dual_specification_effects.py` | `Unified_Visualization/plot_figure_7_rq2_dual_specification_effects_20260723.py` |
| `causal/Unified_Visualization/plot_figure_7_rq2_suning_reversals.py` | `Unified_Visualization/plot_figure_7_rq2_suning_reversals_20260723.py` |
| `causal/Unified_Visualization/plot_figure_7_stage_specific_effects.py` | `Unified_Visualization/plot_figure_7_stage_specific_effects_20260721.py` |
| `causal/Unified_Visualization/plot_figure_8_rq2_dual_specification_iqr_effects.py` | `Unified_Visualization/plot_figure_8_rq2_dual_specification_iqr_effects_20260723.py` |
| `causal/Unified_Visualization/plot_figure_8_suning_iqr_scaled_effects.py` | `Unified_Visualization/plot_figure_8_suning_iqr_scaled_effects_20260721.py` |
| `causal/Unified_Visualization/plot_figure_9_cate_heterogeneity.py` | `Unified_Visualization/plot_figure_9_cate_heterogeneity_20260723.py` |
| `causal/Unified_Visualization/plot_figure_9_effect_modification.py` | `Unified_Visualization/plot_figure_9_effect_modification_20260721.py` |
| `causal/Unified_Visualization/prepare_and_evaluate_mcre_human_validation.py` | `Unified_Visualization/prepare_and_evaluate_mcre_human_validation_20260728.py` |
| `causal/Unified_Visualization/statistic.py` | `Unified_Visualization/statistic.py` |
| `causal/Unified_Visualization/visualization_common.py` | `Unified_Visualization/visualization_common_20260721.py` |
| `causal/amazon_placebo.py` | `amazon_placebo_20260717.py` |
| `causal/baseline/A_base/CIRS/cirs_1_prep_data.py` | `baseline/A_base/CIRS/cirs_1_prep_data_20260717.py` |
| `causal/baseline/A_base/CIRS/cirs_2_model_api.py` | `baseline/A_base/CIRS/cirs_2_model_api_20260717.py` |
| `causal/baseline/A_base/CIRS/cirs_3_train_export.py` | `baseline/A_base/CIRS/cirs_3_train_export_20260717.py` |
| `causal/baseline/A_base/DICE/dice_1_prep_data.py` | `baseline/A_base/DICE/dice_1_prep_data_20260717.py` |
| `causal/baseline/A_base/DICE/dice_2_model_wrapper.py` | `baseline/A_base/DICE/dice_2_model_wrapper_20260717.py` |
| `causal/baseline/A_base/DICE/dice_3_train_export.py` | `baseline/A_base/DICE/dice_3_train_export_20260717.py` |
| `causal/baseline/A_base/KMCLR/KMCLR/kmclr_1_build_data.py` | `baseline/A_base/KMCLR/KMCLR/kmclr_1_build_data_20260717.py` |
| `causal/baseline/A_base/KMCLR/KMCLR/kmclr_2_core_engine.py` | `baseline/A_base/KMCLR/KMCLR/kmclr_2_core_engine_20260717.py` |
| `causal/baseline/A_base/KMCLR/KMCLR/kmclr_3_train_and_export.py` | `baseline/A_base/KMCLR/KMCLR/kmclr_3_train_and_export_20260717.py` |
| `causal/baseline/A_base/MBGCN/mbgcn_1_build_graph_data.py` | `baseline/A_base/MBGCN/mbgcn_1_build_graph_data_20260717.py` |
| `causal/baseline/A_base/MBGCN/mbgcn_2_core_engine.py` | `baseline/A_base/MBGCN/mbgcn_2_core_engine_20260717.py` |
| `causal/baseline/A_base/MBGCN/mbgcn_3_train_and_export.py` | `baseline/A_base/MBGCN/mbgcn_3_train_and_export_20260717.py` |
| `causal/baseline/A_base/MGAT/mgat_1_prep_data.py` | `baseline/A_base/MGAT/mgat_1_prep_data_20260717.py` |
| `causal/baseline/A_base/MGAT/mgat_2_core_engine.py` | `baseline/A_base/MGAT/mgat_2_core_engine_20260717.py` |
| `causal/baseline/A_base/MGAT/mgat_3_train_export.py` | `baseline/A_base/MGAT/mgat_3_train_export_20260717.py` |
| `causal/baseline/A_base/MGCE/mgce_1_prep_data.py` | `baseline/A_base/MGCE/mgce_1_prep_data_20260717.py` |
| `causal/baseline/A_base/MGCE/mgce_2_model_api.py` | `baseline/A_base/MGCE/mgce_2_model_api_20260717.py` |
| `causal/baseline/A_base/MGCE/mgce_3_train_export.py` | `baseline/A_base/MGCE/mgce_3_train_export_20260717.py` |
| `causal/baseline/A_base/MICRO/micro_1_prep_data.py` | `baseline/A_base/MICRO/micro_1_prep_data_20260717.py` |
| `causal/baseline/A_base/MICRO/micro_2_core_engine.py` | `baseline/A_base/MICRO/micro_2_core_engine_20260717.py` |
| `causal/baseline/A_base/MICRO/micro_3_train_export.py` | `baseline/A_base/MICRO/micro_3_train_export_20260717.py` |
| `causal/baseline/A_base/SLMRec/slmrec_1_prep_data.py` | `baseline/A_base/SLMRec/slmrec_1_prep_data_20260717.py` |
| `causal/baseline/A_base/SLMRec/slmrec_2_model_wrapper.py` | `baseline/A_base/SLMRec/slmrec_2_model_wrapper_20260717.py` |
| `causal/baseline/A_base/SLMRec/slmrec_3_train_export.py` | `baseline/A_base/SLMRec/slmrec_3_train_export_20260717.py` |
| `causal/baseline/A_base/baseline_utils.py` | `baseline/A_base/baseline_utils_20260717.py` |
| `causal/baseline/A_base/run_suning_baselines.py` | `baseline/A_base/run_suning_baselines_20260717.py` |
| `causal/baseline/amazon/A_base/CIRS/cirs_1_prep_data.py` | `baseline/amazon/A_base_20260718/CIRS/cirs_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/CIRS/cirs_2_model_api.py` | `baseline/amazon/A_base_20260718/CIRS/cirs_2_model_api_20260718.py` |
| `causal/baseline/amazon/A_base/CIRS/cirs_3_train_export.py` | `baseline/amazon/A_base_20260718/CIRS/cirs_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/DICE/dice_1_prep_data.py` | `baseline/amazon/A_base_20260718/DICE/dice_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/DICE/dice_2_model_wrapper.py` | `baseline/amazon/A_base_20260718/DICE/dice_2_model_wrapper_20260718.py` |
| `causal/baseline/amazon/A_base/DICE/dice_3_train_export.py` | `baseline/amazon/A_base_20260718/DICE/dice_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/KMCLR/kmclr_1_build_data.py` | `baseline/amazon/A_base_20260718/KMCLR/kmclr_1_build_data_20260718.py` |
| `causal/baseline/amazon/A_base/KMCLR/kmclr_2_core_engine.py` | `baseline/amazon/A_base_20260718/KMCLR/kmclr_2_core_engine_20260718.py` |
| `causal/baseline/amazon/A_base/KMCLR/kmclr_3_train_and_export.py` | `baseline/amazon/A_base_20260718/KMCLR/kmclr_3_train_and_export_20260718.py` |
| `causal/baseline/amazon/A_base/MBGCN/mbgcn_1_build_graph_data.py` | `baseline/amazon/A_base_20260718/MBGCN/mbgcn_1_build_graph_data_20260718.py` |
| `causal/baseline/amazon/A_base/MBGCN/mbgcn_2_core_engine.py` | `baseline/amazon/A_base_20260718/MBGCN/mbgcn_2_core_engine_20260718.py` |
| `causal/baseline/amazon/A_base/MBGCN/mbgcn_3_train_and_export.py` | `baseline/amazon/A_base_20260718/MBGCN/mbgcn_3_train_and_export_20260718.py` |
| `causal/baseline/amazon/A_base/MGAT/mgat_1_prep_data.py` | `baseline/amazon/A_base_20260718/MGAT/mgat_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/MGAT/mgat_2_core_engine.py` | `baseline/amazon/A_base_20260718/MGAT/mgat_2_core_engine_20260718.py` |
| `causal/baseline/amazon/A_base/MGAT/mgat_3_train_export.py` | `baseline/amazon/A_base_20260718/MGAT/mgat_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/MGCE/mgce_1_prep_data.py` | `baseline/amazon/A_base_20260718/MGCE/mgce_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/MGCE/mgce_2_model_api.py` | `baseline/amazon/A_base_20260718/MGCE/mgce_2_model_api_20260718.py` |
| `causal/baseline/amazon/A_base/MGCE/mgce_3_train_export.py` | `baseline/amazon/A_base_20260718/MGCE/mgce_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/MICRO/micro_1_prep_data.py` | `baseline/amazon/A_base_20260718/MICRO/micro_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/MICRO/micro_2_core_engine.py` | `baseline/amazon/A_base_20260718/MICRO/micro_2_core_engine_20260718.py` |
| `causal/baseline/amazon/A_base/MICRO/micro_3_train_export.py` | `baseline/amazon/A_base_20260718/MICRO/micro_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/SLMRec/slmrec_1_prep_data.py` | `baseline/amazon/A_base_20260718/SLMRec/slmrec_1_prep_data_20260718.py` |
| `causal/baseline/amazon/A_base/SLMRec/slmrec_2_model_wrapper.py` | `baseline/amazon/A_base_20260718/SLMRec/slmrec_2_model_wrapper_20260718.py` |
| `causal/baseline/amazon/A_base/SLMRec/slmrec_3_train_export.py` | `baseline/amazon/A_base_20260718/SLMRec/slmrec_3_train_export_20260718.py` |
| `causal/baseline/amazon/A_base/amazon_baseline_utils.py` | `baseline/amazon/A_base_20260718/amazon_baseline_utils_20260718.py` |
| `causal/baseline/amazon/dcml_recommender_core_amazon.py` | `baseline/amazon/dcml_recommender_core_amazon_20260718.py` |
| `causal/baseline/amazon/run_amazon_prediction_multiseed.py` | `baseline/amazon/run_amazon_prediction_multiseed_20260718.py` |
| `causal/baseline/amazon/run_amazon_prediction_multiseed_app.py` | `baseline/amazon/run_amazon_prediction_multiseed_app_20260719.py` |
| `causal/baseline/amazon/run_amazon_prediction_multiseed_beauty.py` | `baseline/amazon/run_amazon_prediction_multiseed_beauty_20260719.py` |
| `causal/baseline/amazon/run_prediction_baselines_amazon.py` | `baseline/amazon/run_prediction_baselines_amazon_20260718.py` |
| `causal/baseline/amazon/run_prediction_baselines_amazon_app.py` | `baseline/amazon/run_prediction_baselines_amazon_app_20260718.py` |
| `causal/baseline/amazon/run_prediction_baselines_amazon_beauty.py` | `baseline/amazon/run_prediction_baselines_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/step1_prepare_prediction_baseline_amazon.py` | `baseline/amazon/step1_prepare_prediction_baseline_amazon_20260718.py` |
| `causal/baseline/amazon/step1_prepare_prediction_baseline_amazon_app.py` | `baseline/amazon/step1_prepare_prediction_baseline_amazon_app_20260718.py` |
| `causal/baseline/amazon/step1_prepare_prediction_baseline_amazon_beauty.py` | `baseline/amazon/step1_prepare_prediction_baseline_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/step2_train_dcml_recommender_amazon.py` | `baseline/amazon/step2_train_dcml_recommender_amazon_20260718.py` |
| `causal/baseline/amazon/step2_train_dcml_recommender_amazon_app.py` | `baseline/amazon/step2_train_dcml_recommender_amazon_app_20260718.py` |
| `causal/baseline/amazon/step2_train_dcml_recommender_amazon_beauty.py` | `baseline/amazon/step2_train_dcml_recommender_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/step3_evaluate_dcml_recommender_amazon.py` | `baseline/amazon/step3_evaluate_dcml_recommender_amazon_20260718.py` |
| `causal/baseline/amazon/step3_evaluate_dcml_recommender_amazon_app.py` | `baseline/amazon/step3_evaluate_dcml_recommender_amazon_app_20260718.py` |
| `causal/baseline/amazon/step3_evaluate_dcml_recommender_amazon_beauty.py` | `baseline/amazon/step3_evaluate_dcml_recommender_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/step4_compare_prediction_models_amazon.py` | `baseline/amazon/step4_compare_prediction_models_amazon_20260718.py` |
| `causal/baseline/amazon/step4_compare_prediction_models_amazon_app.py` | `baseline/amazon/step4_compare_prediction_models_amazon_app_20260718.py` |
| `causal/baseline/amazon/step4_compare_prediction_models_amazon_beauty.py` | `baseline/amazon/step4_compare_prediction_models_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/step5_candidate_sensitivity_amazon.py` | `baseline/amazon/step5_candidate_sensitivity_amazon_20260718.py` |
| `causal/baseline/amazon/step5_candidate_sensitivity_amazon_app.py` | `baseline/amazon/step5_candidate_sensitivity_amazon_app_20260718.py` |
| `causal/baseline/amazon/step5_candidate_sensitivity_amazon_beauty.py` | `baseline/amazon/step5_candidate_sensitivity_amazon_beauty_20260718.py` |
| `causal/baseline/amazon/treatment_feature_ablation_amazon_app.py` | `baseline/amazon/treatment_feature_ablation_amazon_app_20260719.py` |
| `causal/baseline/amazon/treatment_feature_ablation_amazon_beauty.py` | `baseline/amazon/treatment_feature_ablation_amazon_beauty_20260719.py` |
| `causal/baseline/prediction_multiseed_supplement.py` | `baseline/prediction_multiseed_supplement_20260719.py` |
| `causal/baseline/summarize_prediction_multiseed.py` | `baseline/summarize_prediction_multiseed_20260720.py` |
| `causal/baseline/suning/dcml_recommender_core.py` | `baseline/suning/dcml_recommender_core_20260717.py` |
| `causal/baseline/suning/run_suning_multiseed.py` | `baseline/suning/run_suning_multiseed_20260719.py` |
| `causal/baseline/suning/step1_prepare_baseline_data.py` | `baseline/suning/step1_prepare_baseline_data_20260717.py` |
| `causal/baseline/suning/step2_train_dcml_recommender.py` | `baseline/suning/step2_train_dcml_recommender_20260717.py` |
| `causal/baseline/suning/step3_4_evaluate_dcml.py` | `baseline/suning/step3_4_evaluate_dcml_20260717.py` |
| `causal/baseline/suning/step4_compare_all_models.py` | `baseline/suning/step4_compare_all_models_20260717.py` |
| `causal/baseline/suning/step5_candidate_sensitivity.py` | `baseline/suning/step5_candidate_sensitivity_20260717.py` |
| `causal/baseline/suning/treatment_feature_ablation.py` | `baseline/suning/treatment_feature_ablation_20260719.py` |
| `causal/build_claim_aligned_robustness.py` | `build_claim_aligned_robustness_20260729_2.py` |
| `causal/channel_analysis_reporting.py` | `channel_analysis_reporting_20260719.py` |
| `causal/data_preprocessing/amazon_appliances_preprocessing/build_y_funnel_amazon.py` | `data_preprocessing/amazon_appliances_preprocessing/build_y_funnel_amazon_20260713.py` |
| `causal/data_preprocessing/amazon_appliances_preprocessing/download_images_amazon_app.py` | `data_preprocessing/amazon_appliances_preprocessing/download_images_amazon_app.py` |
| `causal/data_preprocessing/amazon_appliances_preprocessing/process_confounders.py` | `data_preprocessing/amazon_appliances_preprocessing/process_confounders_20260713.py` |
| `causal/data_preprocessing/amazon_appliances_preprocessing/process_item_reviews_amazon.py` | `data_preprocessing/amazon_appliances_preprocessing/process_item_reviews_amazon_20260713.py` |
| `causal/data_preprocessing/amazon_beauty_preprocessing/download_images_amazon_beauty.py` | `data_preprocessing/amazon_beauty_preprocessing/download_images_amazon_beauty.py` |
| `causal/data_preprocessing/suning_preprocessing/build_y_funnel.py` | `data_preprocessing/suning_preprocessing/build_y_funnel_20260712.py` |
| `causal/data_preprocessing/suning_preprocessing/download_images_suning.py` | `data_preprocessing/suning_preprocessing/download_images_suning.py` |
| `causal/data_preprocessing/suning_preprocessing/process_confounder_user.py` | `data_preprocessing/suning_preprocessing/process_confounder_user.py` |
| `causal/data_preprocessing/suning_preprocessing/process_item_reviews.py` | `data_preprocessing/suning_preprocessing/process_item_reviews.py` |
| `causal/data_preprocessing/suning_preprocessing/process_item_text_and_confounder.py` | `data_preprocessing/suning_preprocessing/process_item_text&confounder.py` |
| `causal/dcml_causal_baselines.py` | `dcml_causal_baselines_20260719.py` |
| `causal/dcml_formal_tests.py` | `dcml_formal_tests_20260717.py` |
| `causal/dcml_gs_sensitivity.py` | `dcml_gs_sensitivity_20260717.py` |
| `causal/dcml_h3_observed_levels.py` | `dcml_h3_observed_levels_20260719.py` |
| `causal/dcml_inference.py` | `dcml_inference_20260717.py` |
| `causal/dcml_nuisance.py` | `dcml_nuisance_20260717.py` |
| `causal/dcml_specification_sensitivity.py` | `dcml_specification_sensitivity_20260719.py` |
| `causal/dcml_utils.py` | `dcml_utils_20260713.py` |
| `causal/dml_ovb_sensitivity.py` | `dml_ovb_sensitivity_20260728.py` |
| `causal/enhance/enhance_common.py` | `enhance/enhance_common_20260730.py` |
| `causal/enhance/run_cross_mllm_inference.py` | `enhance/run_cross_mllm_inference_20260801.py` |
| `causal/enhance/run_cross_mllm_measurement_audit.py` | `enhance/run_cross_mllm_measurement_audit_20260801.py` |
| `causal/enhance/run_efficiency_cost_benchmark.py` | `enhance/run_efficiency_cost_benchmark_20260730.py` |
| `causal/enhance/run_taxonomy_admissibility_sensitivity.py` | `enhance/run_taxonomy_admissibility_sensitivity_20260730.py` |
| `causal/model/M_amazon/DML/causal_baseline_comparison_amazon_app.py` | `model/M_amazon/DML/causal_baseline_comparison_amazon_app_20260719.py` |
| `causal/model/M_amazon/DML/causal_baseline_comparison_amazon_beauty.py` | `model/M_amazon/DML/causal_baseline_comparison_amazon_beauty_20260719.py` |
| `causal/model/M_amazon/DML/causal_baseline_support_consistency_amazon_beauty.py` | `model/M_amazon/DML/causal_baseline_support_consistency_amazon_beauty_20260720.py` |
| `causal/model/M_amazon/DML/dcml_step3_channel_reporting_amazon_app.py` | `model/M_amazon/DML/dcml_step3_channel_reporting_amazon_app_20260719.py` |
| `causal/model/M_amazon/DML/dcml_step3_channel_reporting_amazon_beauty.py` | `model/M_amazon/DML/dcml_step3_channel_reporting_amazon_beauty_20260719.py` |
| `causal/model/M_amazon/DML/dcml_step3_mediation_amazon_app.py` | `model/M_amazon/DML/dcml_step3_mediation_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/dcml_step3_mediation_amazon_beauty.py` | `model/M_amazon/DML/dcml_step3_mediation_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/dml_step1_nuisance_amazon_app.py` | `model/M_amazon/DML/dml_step1_nuisance_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/dml_step1_nuisance_amazon_beauty.py` | `model/M_amazon/DML/dml_step1_nuisance_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/dml_step2_causal_inference_amazon_app.py` | `model/M_amazon/DML/dml_step2_causal_inference_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/dml_step2_causal_inference_amazon_beauty.py` | `model/M_amazon/DML/dml_step2_causal_inference_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/gs_order_sensitivity_amazon_app.py` | `model/M_amazon/DML/gs_order_sensitivity_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/gs_order_sensitivity_amazon_beauty.py` | `model/M_amazon/DML/gs_order_sensitivity_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/h3_moderation_interaction_amazon_app.py` | `model/M_amazon/DML/h3_moderation_interaction_amazon_app_20260719.py` |
| `causal/model/M_amazon/DML/h3_moderation_interaction_amazon_beauty.py` | `model/M_amazon/DML/h3_moderation_interaction_amazon_beauty_20260719.py` |
| `causal/model/M_amazon/DML/omitted_confounding_sensitivity_amazon_app.py` | `model/M_amazon/DML/omitted_confounding_sensitivity_amazon_app_20260728.py` |
| `causal/model/M_amazon/DML/omitted_confounding_sensitivity_amazon_beauty.py` | `model/M_amazon/DML/omitted_confounding_sensitivity_amazon_beauty_20260728.py` |
| `causal/model/M_amazon/DML/placebo_test_amazon_app.py` | `model/M_amazon/DML/placebo_test_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/placebo_test_amazon_beauty.py` | `model/M_amazon/DML/placebo_test_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/robustness_value_amazon_app.py` | `model/M_amazon/DML/robustness_value_amazon_app_20260717.py` |
| `causal/model/M_amazon/DML/robustness_value_amazon_beauty.py` | `model/M_amazon/DML/robustness_value_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/DML/rq2_joint_effect_tests_amazon_app.py` | `model/M_amazon/DML/rq2_joint_effect_tests_amazon_app_20260723.py` |
| `causal/model/M_amazon/DML/rq2_joint_effect_tests_amazon_beauty.py` | `model/M_amazon/DML/rq2_joint_effect_tests_amazon_beauty_20260723.py` |
| `causal/model/M_amazon/DML/specification_sensitivity_amazon_app.py` | `model/M_amazon/DML/specification_sensitivity_amazon_app_20260719.py` |
| `causal/model/M_amazon/DML/specification_sensitivity_amazon_beauty.py` | `model/M_amazon/DML/specification_sensitivity_amazon_beauty_20260719.py` |
| `causal/model/M_amazon/MCRE/amazon_dataset_builder_common.py` | `model/M_amazon/MCRE/amazon_dataset_builder_common_20260717.py` |
| `causal/model/M_amazon/MCRE/dataset_builder_amazon_app.py` | `model/M_amazon/MCRE/dataset_builder_amazon_app_20260717.py` |
| `causal/model/M_amazon/MCRE/dataset_builder_amazon_beauty.py` | `model/M_amazon/MCRE/dataset_builder_amazon_beauty_20260717.py` |
| `causal/model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon.py` | `model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon_20260713.py` |
| `causal/model/M_amazon/MCRE/merge_multimodal_scalars_amazon.py` | `model/M_amazon/MCRE/merge_multimodal_scalars_amazon_20260713.py` |
| `causal/model/M_suning/DML/causal_baseline_comparison.py` | `model/M_suning/DML/causal_baseline_comparison_20260719.py` |
| `causal/model/M_suning/DML/dcml_step1_nuisance_models.py` | `model/M_suning/DML/dcml_step1_nuisance_models_20260717.py` |
| `causal/model/M_suning/DML/dcml_step2_causal_inference.py` | `model/M_suning/DML/dcml_step2_causal_inference_20260717.py` |
| `causal/model/M_suning/DML/dcml_step3_channel_reporting.py` | `model/M_suning/DML/dcml_step3_channel_reporting_20260719.py` |
| `causal/model/M_suning/DML/dcml_step3_mediation.py` | `model/M_suning/DML/dcml_step3_mediation_20260717.py` |
| `causal/model/M_suning/DML/gs_order_sensitivity.py` | `model/M_suning/DML/gs_order_sensitivity_20260717.py` |
| `causal/model/M_suning/DML/h3_moderation_interaction.py` | `model/M_suning/DML/h3_moderation_interaction_20260719.py` |
| `causal/model/M_suning/DML/h4_stage_contrast.py` | `model/M_suning/DML/h4_stage_contrast_20260717.py` |
| `causal/model/M_suning/DML/omitted_confounding_sensitivity.py` | `model/M_suning/DML/omitted_confounding_sensitivity_20260728.py` |
| `causal/model/M_suning/DML/placebo_test.py` | `model/M_suning/DML/placebo_test_20260717.py` |
| `causal/model/M_suning/DML/robustness_value.py` | `model/M_suning/DML/robustness_value_20260717.py` |
| `causal/model/M_suning/DML/rq2_joint_effect_tests.py` | `model/M_suning/DML/rq2_joint_effect_tests_20260723.py` |
| `causal/model/M_suning/DML/specification_sensitivity.py` | `model/M_suning/DML/specification_sensitivity_20260719.py` |
| `causal/model/M_suning/DML/suning_iqr_decision_analysis.py` | `model/M_suning/DML/suning_iqr_decision_analysis_20260720.py` |
| `causal/model/M_suning/MCRE/dataset_builder.py` | `model/M_suning/MCRE/dataset_builder_20260712.py` |
| `causal/model/M_suning/MCRE/extract_lmm_multimodal_feature_attention.py` | `model/M_suning/MCRE/extract_lmm_multimodal_feature_attention.py` |
| `causal/model/audit_mcre_measurement_support.py` | `model/audit_mcre_measurement_support_20260722.py` |
| `causal/model/audit_mcre_temporal_snapshot.py` | `model/audit_mcre_temporal_snapshot_20260717.py` |
| `causal/nuisance_temporal_transport_diagnostics.py` | `nuisance_temporal_transport_diagnostics_20260719.py` |
| `causal/postprocess_dml_ovb_sensitivity.py` | `postprocess_dml_ovb_sensitivity_20260729.py` |
| `causal/recommendation_feature_ablation.py` | `recommendation_feature_ablation_20260719.py` |
| `causal/rq2_joint_effect_tests.py` | `rq2_joint_effect_tests_20260723.py` |
| `causal/summarize_rq5_robustness.py` | `summarize_rq5_robustness_20260729.py` |
| `causal/temporal_certification.py` | `temporal_certification_20260717.py` |
| `causal/temporal_selection_sensitivity.py` | `temporal_selection_sensitivity_20260719.py` |
| `causal/validate_claim_aligned_robustness.py` | `validate_claim_aligned_robustness_20260729_2.py` |
| `causal/validate_rq5_outputs.py` | `validate_rq5_outputs_20260729.py` |
| `causal/validate_supplemental_outputs.py` | `validate_supplemental_outputs_20260719.py` |
| `causal/validate_targeted_corrections.py` | `validate_targeted_corrections_20260720.py` |
