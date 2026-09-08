# Data access, layout, and contracts

## Dataset availability

No raw or processed datasets, product images, human annotations, model weights,
or fitted artifacts are uploaded to this repository. The `data/` directories
contain instructions only; the configuration schema contains field names only.

**Private data:** Suning is a private enterprise dataset. Please contact the authors
to request access. Access to Suning data, private annotations, and other restricted
research artifacts is subject to the data owner's authorization and applicable
agreements. Author names are listed in [CITATION.cff](../CITATION.cff); use the
corresponding-author contact provided with the accompanying manuscript.

**Public data:** download Amazon Appliances and All Beauty from the
[official Amazon Reviews'23 project](https://amazon-reviews-2023.github.io/).
Use the original category review and metadata files listed below.

| Dataset | Reviews (.jsonl.gz) | Item metadata (.jsonl.gz) |
| --- | --- | --- |
| Appliances | [Appliances.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz) | [meta_Appliances.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz) |
| All Beauty | [All_Beauty.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz) | [meta_All_Beauty.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz) |

Decompress all four files. Rename `Appliances.jsonl` to `reviews_Appliances.jsonl`
locally; keep the other three decompressed basenames unchanged. Place them in the
raw-input layout below, or configure an external raw-data directory. Obtain product
images locally with the supplied image download scripts. Cite the dataset provider
and follow its terms of use. If a direct link moves, use the category table on the
official project page to locate the current download.

## Configure locations once

`causal/project_paths.py` reads `configs/paths.local.json` when present. You can use
another file with `DCML_PATH_CONFIG`. Relative paths are resolved from the repository
root. The following environment variables override individual JSON settings:

| JSON key | Default | Environment variable |
| --- | --- | --- |
| `raw_data` | `data/raw` | `DCML_RAW_DATA_DIR` |
| `processed_data` | `data/processed` | `DCML_PROCESSED_DATA_DIR` |
| `results_of_comparison` | `outputs/prediction` | `DCML_PREDICTION_DIR` |
| `log` | `outputs/logs` | `DCML_LOG_DIR` |
| `Unified_Visualization/output` | `outputs/figures` | `DCML_FIGURE_DIR` |
| `enhance/output` | `outputs/enhance` | `DCML_ENHANCE_DIR` |
| `archives` | `archives` | `DCML_ARCHIVE_DIR` |

For example:

```bash
export DCML_PROCESSED_DATA_DIR=/path/to/causal/processed_data
python run.py --paths
```

The legacy `DCML_CAUSAL_ROOT` can point to a complete research workspace containing
`raw_data`, `processed_data`, `results_of_comparison`, and `log`. A specific JSON or
environment setting takes precedence. Explicit `--causal-root` arguments accepted
by individual scripts retain their original meaning and select the supplied tree.
The public `run.py` uses the central configuration.

Dates in data names remain unchanged. Keep the version requested by the script;
“latest file on disk” is not a substitute for a matching experimental contract.
Some audit scripts intentionally inspect older inputs alongside newer results.
Archive lookups retain directories such as `archives/20260728_results`.

## Raw inputs

The Suning inputs are private enterprise data; contact the authors to request access.
No raw records, product media, expert annotations, or fitted model artifacts are
distributed. The code expects
the source workbook schemas used in the study; adapting another dataset requires
mapping that dataset to these fields.

Amazon inputs come from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/).
Download and decompress the Appliances and All Beauty metadata/review JSONL files.
Use the following local names (the Appliances review file is renamed locally):

```text
data/raw/
├── amazon_appliances_raw/
│   ├── meta_Appliances.jsonl
│   └── reviews_Appliances.jsonl
├── amazon_beauty_raw/
│   ├── meta_All_Beauty.jsonl
│   └── All_Beauty.jsonl
└── suning_raw/                  # Only when independently authorized/available
```

Amazon preparation scripts under `amazon_appliances_preprocessing` process both
Amazon datasets, except the image downloader, which has a separate Beauty entry.
The Amazon outcome builder uses review/rating presence as `y_purchase`, together
with sampled negatives. This is a purchase proxy, not a logged checkout outcome.

## Intermediate tables

Within `data/processed/<dataset>/`, the important directories are:

| Directory | Content and key fields |
| --- | --- |
| `Y` | User-item outcomes: `user_id`, `item_id`, `timestamp`, `y_purchase`; Suning also has `y_click`, `y_cart` |
| `Confounder_user` | `user_id` and user covariates produced by preprocessing |
| `item_feature&Confounder_price` | `item_id`, item text, category/metadata, and `x_pri_log` |
| `pure_reviews` | `user_id`, `item_id`, `review_text`, `timestamp` |
| `rating` | `user_id`, `item_id`, rating and temporal information |
| `picture` (Suning) / `pictures` (Amazon) | Product images; Suning extractor uses `<item_id>_1.jpg`, Amazon uses `<item_id>_MAIN.*` |
| `item_multimodal_scalars` | `item_id`, measured composites/atoms, and extraction metadata |
| `build_dataset` | Chronological A/B/C splits and a treatment schema |
| `audit` | Temporal certification summary and per-item status |
| `DML_Results` | Nuisance models, residuals, and feature schemas |
| `Final_Causal_Output` | ATE/CATE, joint tests, support diagnostics, and sensitivities |

Keep identifiers consistent across tables. The scripts determine exact columns
from their schemas and fail if a required field is absent. Full-period Amazon user
aggregates are excluded from the final nuisance covariate whitelist. `H_level` and
`M_norm` are retained for their defined diagnostic roles, not freely added to X.

## Frozen estimation inputs

The final Suning nuisance routine reads:

```text
data/processed/suning/
├── build_dataset/
│   ├── DCML_A_20260712.parquet
│   ├── DCML_B_20260712.parquet
│   ├── DCML_C_20260712.parquet
│   └── DCML_dataset_schema_20260712.json
├── DML_Results/
│   └── DML_feature_schema_20260712.json
└── audit/
    ├── mcre_temporal_snapshot_summary_20260717.json
    └── mcre_temporal_snapshot_item_certification_20260717.csv
```

The original Suning builder does not recreate the earlier feature-whitelist file.
`configs/suning_feature_schema.json` contains that frozen structural schema without
any observations. Original category identifiers use JSON Unicode escapes to preserve
their exact decoded values in the English source release. `python tools/prepare_workspace.py`
installs it under the expected artifact name only when the destination does not
already exist. Adapt the whitelist explicitly when studying a different dataset.

For `amazon_appliances` or `amazon_beauty`, replace `<dataset>` below:

```text
build_dataset/<dataset>_DCML_Set_A_20260717.parquet
build_dataset/<dataset>_DCML_Set_B_20260717.parquet
build_dataset/<dataset>_DCML_Set_C_20260717.parquet
build_dataset/<dataset>_DCML_schema_20260717.json
audit/mcre_temporal_snapshot_summary_20260717.json
audit/mcre_temporal_snapshot_item_certification_20260717.csv
```

The assembled splits need `user_id`, `item_id`, `timestamp`, `H_level`, `M_norm`,
the 12 measurement treatments, dataset outcomes, and the declared covariates.
Composite-6 and component-10 models remain distinct. The nuisance stage fits on A
and evaluates on B for GS training residuals; it fits on B and evaluates on C for
final residuals. Downstream inference uses the matching `20260717` schemas.

CSV fallback is supported by many shared readers but not every direct Parquet
reader. Parquet is recommended for the complete pipeline. The source contract,
not a generic example CSV, determines the required input format.

## Referenced assets in manifests

Human-validation and cross-model audit tables may contain image paths. On a new
machine those fields must refer to locally available images; changing only the
directory configuration cannot repair arbitrary old absolute paths stored inside
a CSV. Regenerate the input inventory with the supplied prepare command, or update
its paths while keeping item identities and audit design fixed.
