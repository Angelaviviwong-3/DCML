# DCML

**Multimodal treatment measurement and double machine learning for e-commerce recommendation.**

Research code accompanying:

> Huan Wang, Yongdong Shi, Chunyan Fan, and Shuixia Chen. *From Multimodal Cues to Funnel Outcomes: MLLM-Based Treatment Measurement and Double Machine Learning in E-Commerce Recommendation.*

[Run order](docs/RUN_ORDER.md) · [Data contracts](docs/DATA.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Source inventory](docs/SOURCE_VERSIONS.md)

DCML measures multimodal product cues with Qwen2-VL, constructs temporally ordered
analysis samples, fits nuisance models on separate time splits, and estimates
support-qualified effects under composite and component treatment specifications.
The repository also contains measurement validation, heterogeneity and sensitivity
analyses, recommendation baselines, and cross-MLLM audits.

## Quick start

Use Python 3.10–3.12; the CPU release checks were run with Python 3.12. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_release.py
python examples/synthetic_demo.py
python run.py --list --dataset suning
```

On Windows, activate with `.venv\Scripts\activate`. The demo generates artificial
residual data and exercises the retained projection, support checks, and effect
estimation code. It requires no private data, GPU, API key, or model download.
Its results are software examples, not results from the paper.

For Qwen extraction, install a suitable PyTorch/CUDA build and then
`requirements-mllm.txt`. Recommendation models use `requirements-baselines.txt`.
Optional GPU environments are described in [Reproducibility](docs/REPRODUCIBILITY.md).

## Repository layout

```text
DCML/
├── causal/
│   ├── data_preprocessing/       # Suning and Amazon preparation
│   ├── model/                   # MCRE, DML, and dataset entry points
│   ├── baseline/                # Final recommendation rerun implementations
│   ├── Unified_Visualization/   # Measurement validation and figures
│   ├── enhance/                 # Taxonomy, cross-MLLM, and efficiency audits
│   ├── dcml_*.py                # Shared estimation routines
│   └── project_paths.py         # Central filesystem configuration
├── configs/                     # Portable path and schema settings
├── data/{raw,processed}/         # User-supplied data; excluded from Git
├── outputs/                     # Generated results; excluded from Git
├── examples/                    # Synthetic CPU example
├── docs/                        # Reproduction and source-version notes
├── tools/                       # Workspace setup and release checks
├── tests/                       # Portability and numerical smoke checks
└── run.py                       # Dataset-stage launcher
```

## Paths and versioned artifacts

Code files and code directories have clean names without date suffixes. Dates in
data filenames, experiment constants, and output tags are retained because they
identify frozen experimental contracts. For example, `dcml_step1_nuisance_models.py`
still reads `DCML_B_20260712.parquet` and writes the `20260717` nuisance results.
Do not rename an old artifact to make it appear to satisfy a newer contract.

Default paths are relative to this repository, regardless of the working
directory. To keep data on another disk:

```bash
cp configs/paths.example.json configs/paths.local.json
```

Edit only the needed values in `paths.local.json`, for example:

```json
{
  "raw_data": "/path/to/data/raw",
  "processed_data": "/path/to/data/processed",
  "results_of_comparison": "outputs/prediction"
}
```

Check the effective paths with `python run.py --paths`. The local configuration is
ignored by Git. Environment overrides and existing research-workspace layouts
are documented in [Data contracts](docs/DATA.md).

## Running the research pipeline

Prepare the input data and frozen schema, then follow [the full run order](docs/RUN_ORDER.md).
The launcher runs one explicit stage at a time:

```bash
python tools/prepare_workspace.py
python run.py --stage nuisance --dataset suning --dry-run
python run.py --stage nuisance --dataset suning
python run.py --stage inference --dataset suning
```

The same launcher supports `amazon_appliances` and `amazon_beauty`. Dataset assembly,
temporal certification, human annotation, estimation, and prediction are distinct
steps; the launcher does not silently skip their prerequisites.

## Data and interpretation

**No datasets are included in this repository.**

| Dataset | Access |
| --- | --- |
| Amazon Appliances | Public: [reviews](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz) and [item metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz) |
| Amazon All Beauty | Public: [reviews](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz) and [item metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz) |
| Suning | Private: please contact the authors to request access, subject to the data owner's authorization. |

The public download links come from the [Amazon Reviews'23 project](https://amazon-reviews-2023.github.io/).
See [Data access and preparation](docs/DATA.md) for decompression, local filenames,
and required schemas. Private annotations and other restricted research artifacts
are also available only by contacting the authors, subject to applicable permissions.
This release contains code and schemas, not the original datasets, product images,
model weights, fitted models, human ratings, or frozen result archives.

Suning has click, cart, and purchase outcomes. The Amazon branch uses review/rating
records as a purchase-proxy task with sampled negatives; it does not contain native
click or cart events. Causal interpretations remain conditional on the manuscript's
observational identification and temporal-selection assumptions. Offline prediction
metrics and proxy-channel diagnostics are separate from causal effect estimates.

The public documentation, comments, messages, and Suning extraction prompt are in
English. Original dataset field identifiers remain compatible. Translating the
Suning prompt and generated text labels can change newly extracted model scores;
see [English publication changes](docs/REPRODUCIBILITY.md#english-publication-changes).

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff); publication details
can be added when available. The authors' code is provided under the [MIT License](LICENSE),
with baseline-specific licenses retained as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
