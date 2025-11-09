## Benchmark Data Download and Directory Organization Instructions

This directory is used to store the raw data files for various benchmarks (datasets). Please download and place the corresponding data files according to the current subdirectory structure of this project, so that scripts and evaluation tools can directly locate and use the data.

### Current Directory Structure (Subdirectories) Example:

```
benchmark/
├── comparisonqa/
├── hotpotqa/
├── math/
├── math_perturb/
├── math500/
├── strategyqa/
└── triviaqa/
```


Each subdirectory should contain the **original data files** (typically in `.json` format) for its respective benchmark.

---

### How to Download

- Please download the data from the **official repository** or an **authoritative mirror** of each benchmark.
- For quick start, `strategyqa` and `math_perturb` are provided as examples, both cloned from their official sources.
- If you already have data packages (e.g., `hotpotqa.zip`, `triviaqa.jsonl`, etc.), please **extract them** and **move or copy the original files** into the corresponding subdirectory in this repository.
