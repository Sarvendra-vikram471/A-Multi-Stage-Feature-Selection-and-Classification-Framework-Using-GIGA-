# GIGA: Gini-Guided Genetic Algorithm for Software Defect Prediction

## Overview

This repository contains the implementation of **GIGA** — a novel metaheuristic algorithm for **feature selection in software defect prediction**. GIGA combines Gini impurity-guided initialization with elitism-based local refinement on top of a standard Genetic Algorithm (GA) to achieve superior defect classification performance.

GIGA is evaluated against three baseline metaheuristics:
- **GWO** — Grey Wolf Optimizer
- **WHO** — Wild Horse Optimizer
- **FHO** — Fire Hawk Optimizer
- **GA** — Standard Genetic Algorithm (ablation baseline)

---

## Repository Structure

```
GIGA_GitHub_Submission/
│
├── main.py                  # Main experiment: defect prediction with all algorithms
├── cross_validation.py      # 5-fold & 10-fold stratified cross-validation
├── ablation_study.py        # Convergence plots & ablation analysis (GIGA variants)
├── runtime_benchmark.py     # Wall-clock runtime comparison across all algorithms
├── generate_report.py       # Generates complexity & runtime Word report (.docx)
│
├── datasets/                # 10 benchmark software defect datasets (PROMISE repo)
│   ├── data_ant.csv
│   ├── data_camel.csv
│   ├── data_ivy.csv
│   ├── data_lucene.csv
│   ├── bTomcat.csv
│   ├── bXalan.csv
│   ├── bXerces.csv
│   ├── bprop-6.csv
│   ├── jedit-4.3.csv
│   └── poi-3.0.csv
│
└── results/                 # Pre-computed experimental results
    ├── Defect_Prediction_Results.xlsx   # Main experiment (Accuracy & F1)
    ├── CV_Results.xlsx                  # Cross-validation results
    ├── Ablation_Results.xlsx            # Ablation study results
    ├── runtime_comparison.csv           # Runtime data (CSV)
    └── runtime_comparison.xlsx          # Runtime data (styled Excel)
```

---

## Datasets

All datasets are from the **PROMISE Software Engineering Repository** and cover real-world Java open-source projects. Each dataset contains CK (Chidamber & Kemerer) object-oriented metrics as features and a binary defect label.

| Dataset     | Project      | # Instances |
|-------------|--------------|-------------|
| data_ant    | Apache Ant   | ~745        |
| data_camel  | Apache Camel | ~965        |
| data_ivy    | Apache Ivy   | ~352        |
| data_lucene | Lucene       | ~340        |
| bTomcat     | Tomcat       | ~858        |
| bXalan      | Xalan        | ~909        |
| bXerces     | Xerces       | ~588        |
| bprop-6     | prop-6       | ~660        |
| jedit-4.3   | JEdit 4.3    | ~492        |
| poi-3.0     | POI 3.0      | ~442        |

---

## Algorithms

### GIGA (Proposed)
- **Stage 1**: Gini impurity-guided chromosome initialization — features with higher Gini importance are more likely to be included in the initial population.
- **Stage 2**: Standard GA evolution with tournament selection, single-point crossover, and bit-flip mutation.
- **Stage 3**: Elitism-based local refinement — elite chromosomes undergo single-bit perturbations to escape local optima.

### Fitness Function
```
fitness = 0.9 × (1 − F1-score) + 0.1 × (selected_features / total_features)
```
Balances classification accuracy with feature subset minimization.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Main Experiment (Defect Prediction)

Runs GWO, WHO, FHO, GA, and GIGA on all 10 datasets and saves results to `Defect_Prediction_Results.xlsx`.

```bash
python main.py
```

### 3. Cross-Validation

Runs 5-fold and 10-fold stratified cross-validation for all algorithms.

```bash
python cross_validation.py
```

### 4. Ablation Study

Generates convergence plots and ablation results comparing GIGA variants.

```bash
python ablation_study.py
```

### 5. Runtime Benchmark

Measures wall-clock execution time for each algorithm on each dataset.

```bash
python runtime_benchmark.py
```

### 6. Report Generation

Generates a Word document with complexity and runtime analysis.

```bash
python generate_report.py
```

---

## Results Summary

All pre-computed results are in the `results/` folder. Key findings:

- **GIGA consistently achieves the highest F1-score** across all 10 datasets compared to GWO, WHO, FHO, and GA.
- **Feature reduction**: GIGA selects a compact, informative feature subset, improving model interpretability.
- **Convergence**: GIGA converges faster and to a better solution than standard GA, as shown in the ablation study.
- **Runtime**: GIGA adds modest overhead compared to GA due to Gini initialization, which remains acceptable across all datasets.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

**Core libraries:**
- `numpy`, `pandas` — data handling
- `scikit-learn` — classifiers, cross-validation, metrics
- `xgboost` — XGBoost classifier (used in baselines)
- `matplotlib` — convergence and ablation plots
- `openpyxl` — styled Excel output
- `python-docx` — Word report generation

---

## Citation / Reference

If you use this code or datasets, please cite the PROMISE repository:

> Jureczko, M., & Madeyski, L. (2010). *Towards identifying software project clusters with regard to defect prediction*. PROMISE '10.

---

## Contact

For questions regarding this capstone project, please open an issue or contact the repository maintainer.
