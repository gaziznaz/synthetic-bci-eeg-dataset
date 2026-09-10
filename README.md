# BCI Cognitive Monitoring System — synthetic data, generator, and analysis code

This repository accompanies the article *"A Simulation-Based Rule Architecture
for Real-Time Cognitive State Assessment in BCI Educational Systems"*. It
contains everything needed to reproduce every quantitative result reported in
the article: the exact synthetic dataset, the generator that produced it, the
rule-based cognitive-state analyzer, and the analysis scripts for the
internal-consistency check, the timing benchmark, the machine-learning
comparison, and the individual-baseline calibration.

No real EEG recordings or human participants are involved anywhere in this
repository. All data are synthetically generated.

## Contents

```
.
├── synthetic_eeg_generator.py     # Generates the synthetic EEG dataset
├── cognitive_analyzer.py          # Rule-based CognitiveStateAnalyzer (Attention/Workload/Drowsiness, Stable/Overloaded/Critical)
├── data/
│   ├── all_students_sessions_new.csv   # The exact dataset used for every number in the article
│   └── generator_metadata.json         # Full generation parameters (seed, per-student profiles, scaling factors, phase schedules)
└── analysis/
    ├── consistency_check.py       # Internal consistency: rule-based state vs. generator's programmed phase (97.3%)
    ├── timing_benchmark.py        # Per-call processing time and UI-refresh headroom
    ├── ml_comparison.py           # Logistic-regression sanity check (student-grouped 5-fold CV)
    └── baseline_calibration_stats.py  # Individual-baseline (first 3 min) delta statistics and correlation
```

## Requirements

Python ≥ 3.10, with:

```
numpy
pandas
scipy
scikit-learn
```

## Reproducing the exact dataset

`data/all_students_sessions_new.csv` is already the exact file used
throughout the article. It was produced by:

```bash
python synthetic_eeg_generator.py --seed 773043837954592784
```

This has been re-run and verified to produce a file that is **byte-for-byte
identical** (MD5 checksum match) to `data/all_students_sessions_new.csv`.
Running the generator with a different seed, or without `--seed`, will
produce a new synthetic dataset with the same generative structure but
different random draws — useful for checking that the reported results are
not an artifact of one particular draw, but not identical to the published
numbers.

`data/generator_metadata.json` records the full internal state of that run:
the seed, every student's assigned profile and phase schedule, per-student
band/channel scaling factors, and the software versions used
(Python 3.13.5, NumPy 2.4.6, pandas 3.0.3).

## Reproducing the article's numbers

Run each analysis script from the `analysis/` directory (or anywhere — the
paths are resolved relative to the script location, not the working
directory):

```bash
cd analysis
python consistency_check.py          # -> 97.3% overall match (Section 3)
python timing_benchmark.py           # -> per-call time and headroom table (Section 2.4)
python ml_comparison.py              # -> 78.3%/87.0% accuracy figures (Section 3)
python baseline_calibration_stats.py # -> individual-baseline deltas and r = -0.57 (Section 3)
```

Each script prints the exact figures quoted in the article. Note that the
absolute numbers in `timing_benchmark.py` (milliseconds per call) are
hardware-dependent and will differ from machine to machine — what should
reproduce consistently is the *relative* headroom versus the interface's
3000 ms refresh budget.

## What is intentionally not included

Exploratory analyses that were tried during development but not used in the
article (e.g. an early 3-class distribution summary, and a first, since
corrected, version of the calibration script that clipped scores to a
[0, 100] floor and produced an artificial 0.0 ± 0.0 saturation for one
profile) are not included here, to avoid presenting anything inconsistent
with the published results. `baseline_calibration_stats.py` is the corrected
version that produces the numbers actually reported in the article.

The student/teacher visualization interface is also not included, as it does
not affect any of the reported quantitative results.

## Citing

If you use this code or dataset, please cite the article (full reference to
be added upon publication).
