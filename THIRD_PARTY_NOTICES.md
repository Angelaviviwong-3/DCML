# Baseline implementations and upstream notices

The baseline folders contain the study's retained rerun implementations of
MBGCN, KMCLR, MGAT, MICRO, SLMRec, DICE, CIRS, and MGCE. These are the implementations
used by this research workflow, including its common splits, candidate sets,
training controls, and evaluation protocol. They should not be represented as
unmodified upstream repositories or as universal reference implementations.

The original workspace contained complete upstream checkouts as well as these
rerun scripts. The publication package includes the rerun scripts and their
shared utilities. Unused upstream source trees, compiled evaluators, pretrained
weights, and bundled datasets are excluded.

Existing license texts have been retained unchanged in both the Suning and Amazon
baseline directories:

| Directory | Retained license and notice |
| --- | --- |
| `DICE` | MIT; copyright 2022 FIB LAB, Tsinghua University |
| `MBGCN` | MIT; copyright 2022 FIB LAB, Tsinghua University |
| `MICRO` | MIT; copyright 2021 Big Data and Multi-modal Computing Group, CRIPAC |
| `SLMRec` | GNU General Public License v3; upstream README credits Communication University of China |

These per-directory licenses take precedence over the root MIT license. In
particular, the SLMRec folders retain their GPL license. No additional license
text was present in the source workspace for CIRS, KMCLR, MGAT, or MGCE; no license
claim is made here about the excluded upstream repositories. The root license
covers the authors' contributions, subject to any underlying third-party rights.

Please cite the original baseline papers as well as the accompanying DCML
manuscript when using the comparison code. Dependency packages retain their own
licenses. Downloaded Qwen model weights and Amazon data are governed by their
respective providers' terms.
