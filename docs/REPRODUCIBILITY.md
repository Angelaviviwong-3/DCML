# Reproducibility and environment

## Scope

The release preserves the selected research code's estimators, thresholds, seeds,
split rules, support requirements, and artifact tags. Publication changes concern
filenames, module references, path resolution, English text, and release checks.
The Suning extraction prompt is translated into English as described below.
The original research workspace is not part of the public package.

## English publication changes

Documentation, code comments, command-line help, progress messages, and logs are
in English. The Suning extraction prompt and generated product-text labels are
also translated. The scoring rubric, JSON keys, sampling settings, and numerical
aggregation routines are retained, but prompt translation can change MLLM output.
Fresh extraction with this release is therefore an English-prompt measurement run,
not a claim of exact reproduction of historical Chinese-prompt scores. Keep newly
generated measurements in a separate configured data directory when comparing runs.

Source-data header mappings, empty-review filters, the no-spending category, and
the frozen feature whitelist preserve their original decoded values using Unicode
escapes. These are schema and parsing constants, not observations. Product-category
parsing accepts both English `Category:` labels and the original Chinese label.
Raw product names and review content keep the language of the dataset.

The original-source hashes in `source_manifest.json` identify the selected research
versions before publication edits. They are not checksums of the translated files.

The CPU dependency versions in `requirements.txt` were used for release validation
under Python 3.12. They are a tested publication environment, not a recovered lock
file of the original research server. No complete server environment export was
present in the source directory. Optional MLLM and baseline requirements describe
the libraries imported by those scripts and have not been validated by a full GPU
rerun during packaging.

## Installation details

Install the CPU requirements first. LightGBM needs a working OpenMP runtime. On
macOS, follow the [official LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
if importing `lightgbm` reports a missing `libomp` library. On the packaging machine,
LightGBM was checked using the OpenMP runtime bundled with scikit-learn.

The original nuisance routine has a scikit-learn fallback when LightGBM cannot be
imported. That fallback is retained, but it is a different learner: verify a working
LightGBM installation for comparisons to the historical LightGBM run. Use
`python tools/check_environment.py --group cpu` to check imports.

For Qwen2-VL extraction, use `requirements-mllm.txt` with a PyTorch build appropriate
for the available hardware. `DCML_MLLM_MODEL` may name a locally downloaded model
directory or a Hugging Face model ID; the default remains `Qwen/Qwen2-VL-7B-Instruct`.
Changing that model changes the measurement run. Model files are not bundled.

The recommendation reruns need PyTorch, PyTorch Geometric for MGAT, and TensorFlow
compatibility-v1 operations for CIRS/MGCE. Install `requirements-baselines.txt` in an
environment suitable for those frameworks. The complete multiseed comparison needs
all requested models. For a subset, use the baseline runner's `--models` option.
Use the original seeds for historical comparisons.

## What the release checks cover

- Syntax of every published Python file, clean code names, and resolvable local
  import symbols and literal script references.
- The dataset launcher's complete stage mapping for all three datasets.
- Default, configured, and environment-overridden data/output paths, including
  execution from a different working directory and a relocated repository.
- The final Suning and Amazon baseline runner file inventories and version guards.
- Core CPU module imports and selected command-line help/preflight operations.
- A synthetic example through the retained Gram–Schmidt projection, support gate,
  clustered regression, and bootstrap routines.
- Equality of selected original and packaged core numerical routines on the same
  synthetic input, checked during packaging without importing the source workspace.

The report `VALIDATION.md` records the actual release-check results. Run the
repeatable local checks with:

```bash
python tools/check_release.py
python -m unittest discover -s tests -v
python examples/synthetic_demo.py
```

The original data, human scoring, complete GPU extraction, recommendation training,
and full research sensitivity runs were not rerun as part of packaging. A passing
software check does not reproduce the paper's estimates or establish causal validity.

## Source selection

`source_manifest.json` maps every retained original script to its public filename
and records its original SHA-256. `excluded_sources.json` lists superseded and
excluded source scripts with reasons. Artifact suffixes are intentionally preserved.
The original `config.yaml` was an unused outline; it is replaced by the path
configuration actually read by the published code. Research hyperparameters remain
in their original scripts.

The directory conventions were informed by the separation of code, documentation,
examples, dependencies, and licensing in [EconML](https://github.com/py-why/EconML)
and [CausalML](https://github.com/uber/causalml). No code was copied from those projects.
