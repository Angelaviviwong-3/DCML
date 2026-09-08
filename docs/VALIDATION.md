# Release validation

Validated on 2026-09-09 using Python 3.12.14 on macOS ARM64 in an isolated
publication-check environment.

| Check | Result |
| --- | --- |
| Python syntax | 201 files passed |
| Retained research scripts | 194 source-manifest entries resolved |
| Local import symbols | 866 checked; none missing |
| Literal script references | 138 checked; none missing |
| Dataset-stage launcher | 33 entries resolved |
| Unit checks | 8 passed, including path portability, synthetic inference, and English/original category labels |
| CPU dependency imports | All 13 required imports passed, including LightGBM 4.5.0 |
| Baseline inventories | Suning complete-file preflight and both Amazon preparation preflights passed |
| Synthetic example | Six supported estimates; known effect 0.700, recovered effect 0.693160 |
| Frozen Suning schema | Decoded JSON exactly matches the supplied source schema |
| Input compatibility | Original workbook headers, empty-review filters, and no-spending value retained |
| Documentation links | All relative Markdown file links resolve |
| English publication | No Chinese text remains in published source or documentation; required original data literals use Unicode escapes |
| Publication inventory | 232 source, configuration, documentation, license, and workflow files; no datasets, media, model weights, or result files |
| Credential/path scan | No matched service tokens, private-key blocks, or personal filesystem roots |
| Supplied DCML directory | All 233 release files unchanged by SHA-256; operating-system metadata excluded |

The supplied ZIP matched the local DCML release file for file before English
publication edits. Its original SHA-256 was
`33a78506940f1874ce00779554a61530b69d72db0ce204dfe981f314b07a423e`.
The refreshed publication archive is accompanied by its own SHA-256 checksum.

On this macOS validation machine, LightGBM required the OpenMP runtime bundled
with scikit-learn, supplied through `DYLD_LIBRARY_PATH` for the check process.
No system library or other project's environment was changed. See
[the environment guide](REPRODUCIBILITY.md#installation-details) for installation.

The synthetic result is a software check with artificial data, not a reproduction
of an empirical result. Original observations, human scoring, full GPU extraction,
recommendation training, and complete sensitivity experiments were not rerun.
The Suning prompt is translated to English; newly extracted measurements can differ
from the historical Chinese-prompt run. See the
[English publication notes](REPRODUCIBILITY.md#english-publication-changes).

Repeat the public checks with:

```bash
python tools/check_environment.py --group cpu
python tools/check_release.py
python -m unittest discover -s tests -v
python examples/synthetic_demo.py
```

The included GitHub Actions workflow runs these checks on pushes and pull requests.
Remote workflow status is available in the repository's Actions tab.
