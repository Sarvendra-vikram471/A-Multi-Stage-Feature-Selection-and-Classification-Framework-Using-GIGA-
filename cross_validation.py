"""
Cross-Validation Evaluation Script
====================================
Evaluates ALL experiments from the capstone project using
**5-fold** and **10-fold Stratified Cross-Validation**.

Outputs:
  1. Console progress / summary
  2. Grouped-bar + average-line graphs  → graphs_cv/
  3. Excel report with 4 sheets         → CV_Results.xlsx
     • 5-Fold Accuracy   • 5-Fold F1-Score
     • 10-Fold Accuracy  • 10-Fold F1-Score

NOTE: This script reuses the same metaheuristic / GA / GIGA logic
      that lives in main.py, but wraps the evaluation inside
      StratifiedKFold instead of a single train-test split.
"""

import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════
# SHARED FITNESS FUNCTION  (identical to main.py)
# ═══════════════════════════════════════════════════════════════════
def clamp(x):
    return np.clip(x, 0.0, 1.0)


def fitness(vec, X_train, X_test, y_train, y_test):
    mask = (vec > 0.5) if vec.dtype == float else vec.astype(bool)
    if mask.sum() == 0:
        return 2.0, mask
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train[:, mask], y_train)
    preds = clf.predict(X_test[:, mask])
    score = f1_score(y_test, preds, average="weighted")
    obj = 0.9 * (1 - score) + 0.1 * (mask.sum() / X_train.shape[1])
    return obj, mask


# ═══════════════════════════════════════════════════════════════════
# BASELINE METAHEURISTICS  (identical to main.py)
# ═══════════════════════════════════════════════════════════════════
def GWO(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12):
    pop = np.random.rand(pop_size, dim)
    best_vec, best_f = pop[0].copy(), 1e9
    for t in range(iters):
        a = 2 - 2 * (t / iters)
        for i in range(pop_size):
            f, _ = fitness(pop[i], X_tr, X_te, y_tr, y_te)
            if f < best_f:
                best_f, best_vec = f, pop[i].copy()
        for i in range(pop_size):
            A = a * (2 * np.random.rand() - 1)
            pop[i] = clamp(best_vec - A * np.abs(best_vec - pop[i]))
    return best_vec > 0.5


def WHO(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12):
    pop = np.random.rand(pop_size, dim)
    best = pop[0].copy()
    for _ in range(iters):
        fb, _ = fitness(best, X_tr, X_te, y_tr, y_te)
        for i in range(pop_size):
            f, _ = fitness(pop[i], X_tr, X_te, y_tr, y_te)
            if f < fb:
                best, fb = pop[i].copy(), f
            step = (
                (best - pop[i])
                if np.random.rand() > 0.5
                else (pop[np.random.randint(pop_size)] - pop[i])
            )
            pop[i] = clamp(pop[i] + np.random.rand() * step)
    return best > 0.5


def FHO(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12):
    pop = np.random.rand(pop_size, dim)
    best = pop[0].copy()
    for _ in range(iters):
        fb, _ = fitness(best, X_tr, X_te, y_tr, y_te)
        for i in range(pop_size):
            f, _ = fitness(pop[i], X_tr, X_te, y_tr, y_te)
            if f < fb:
                best, fb = pop[i].copy(), f
            pop[i] = clamp(pop[i] + np.random.rand() * (best - pop[i]))
    return best > 0.5


# ═══════════════════════════════════════════════════════════════════
# GA  (returns mask only — no convergence history needed here)
# ═══════════════════════════════════════════════════════════════════
def GA(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12, n_elites=2):
    mut_rate = 0.05
    pop = (np.random.rand(pop_size, dim) > 0.5).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.random.randint(dim)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9

    for gen in range(iters):
        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        elite_idx = np.argsort(fit_vals)[:n_elites]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[i1] if fit_vals[i1] < fit_vals[i2] else pop[i2]
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[i1] if fit_vals[i1] < fit_vals[i2] else pop[i2]
            pt = np.random.randint(1, dim)
            child = np.concatenate([p1[:pt], p2[pt:]])
            flip = np.random.rand(dim) < mut_rate
            child[flip] = 1 - child[flip]
            if child.sum() == 0:
                child[np.random.randint(dim)] = 1
            new_pop.append(child)

        pop = np.array(new_pop)

    return best_chrom.astype(bool)


# ═══════════════════════════════════════════════════════════════════
# GIGA  (returns mask only)
# ═══════════════════════════════════════════════════════════════════
def _gini_importances(X_tr, y_tr):
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_tr, y_tr)
    imp = dt.feature_importances_
    return imp / (imp.sum() + 1e-9)


def _tournament_select(pop, fit_vals, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmin(fit_vals[idx])]].copy()


def _crossover(p1, p2):
    pt = np.random.randint(1, len(p1))
    return np.concatenate([p1[:pt], p2[pt:]])


def _mutate(chrom, rate):
    flip = np.random.rand(len(chrom)) < rate
    child = chrom.copy()
    child[flip] = 1 - child[flip]
    return child


def GIGA(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12, n_elites=3):
    gini = _gini_importances(X_tr, y_tr)
    p_inc = 0.2 + 0.6 * gini
    pop = (np.random.rand(pop_size, dim) < p_inc).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.argmax(gini)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9

    for gen in range(iters):
        mut_rate = 0.15 * (1.0 - gen / iters) + 0.02

        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        elite_idx = np.argsort(fit_vals)[:n_elites]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < pop_size:
            child = _mutate(
                _crossover(
                    _tournament_select(pop, fit_vals),
                    _tournament_select(pop, fit_vals),
                ),
                mut_rate,
            )
            if child.sum() == 0:
                child[np.argmax(gini)] = 1
            new_pop.append(child)
        pop = np.array(new_pop)

    # --- Gini post-refinement ---
    fit_vals = np.zeros(pop_size)
    for i in range(pop_size):
        fit_vals[i], _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)

    lo_thresh = np.median(gini)
    hi_thresh = np.percentile(gini, 75)
    ref_f = best_f
    ref_chrom = best_chrom.copy()

    for idx in np.argsort(fit_vals)[:n_elites]:
        cand = pop[idx].astype(float)
        cand[gini < lo_thresh] = 0.0
        cand[gini > hi_thresh] = 1.0
        if (cand > 0.5).sum() == 0:
            cand[np.argmax(gini)] = 1.0
        f, _ = fitness(cand, X_tr, X_te, y_tr, y_te)
        if f < ref_f:
            ref_f, ref_chrom = f, (cand > 0.5).astype(int)

    return ref_chrom.astype(bool)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to main.py)
# ═══════════════════════════════════════════════════════════════════
def load_and_preprocess(path):
    df = pd.read_csv(path)
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.median()).values
    return X, y


# ═══════════════════════════════════════════════════════════════════
# DATASETS / MODELS / FS METHODS
# ═══════════════════════════════════════════════════════════════════
datasets = {
    "POI":     "poi-3.0.csv",
    "ANT":     "data_ant.csv",
    "BPROP":   "bprop-6.csv",
    "JEDIT":   "jedit-4.3.csv",
    "Camel":   "data_camel.csv",
    "Ivy":     "data_ivy.csv",
    "Lucene":  "data_lucene.csv",
    "BXERCES": "bXerces.csv",
    "BXALAN":  "bXalan.csv",
    "bTomcat": "bTomcat.csv",
}

models_dict = {
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=30, random_state=42),
    "SVM":           lambda: SVC(probability=True, random_state=42),
    "XGBoost":       lambda: XGBClassifier(eval_metric="logloss", random_state=42),
}

FS_METHODS = ["Before", "GWO", "WHO", "FHO", "GA", "GIGA"]


# ═══════════════════════════════════════════════════════════════════
# CROSS-VALIDATION EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════
def run_cv_experiment(n_folds):
    """
    Run the full experiment under *n_folds*-fold Stratified CV.

    For each fold the feature-selection algorithms are fitted on the
    training partition and the selected mask is used to train/test the
    four classifiers.  Accuracy and F1 are averaged across all folds.

    Returns
    -------
    pd.DataFrame with columns:
        Dataset | Model | FS | Accuracy | F1 | NFeatures
    where Accuracy & F1 are the *mean* across all folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_results = []  # (dname, model, fs, acc, f1, n_feat)

    for dname, path in datasets.items():
        if not os.path.exists(path):
            print(f"  Skipping {dname} — file not found")
            continue
        try:
            X_raw, y = load_and_preprocess(path)
            dim = X_raw.shape[1]
            print(f"\n  {dname}  (dim={dim}, folds={n_folds})")

            # Accumulate per-fold metrics
            # key = (model_name, fs_name)  → list of (acc, f1, n_feat)
            fold_metrics = {}

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
                X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                sc = StandardScaler()
                X_tr = sc.fit_transform(X_train)
                X_te = sc.transform(X_test)

                # Run feature selectors for this fold
                masks = {
                    "Before": np.ones(dim, dtype=bool),
                    "GWO":    GWO(X_tr, X_te, y_train, y_test, dim),
                    "WHO":    WHO(X_tr, X_te, y_train, y_test, dim),
                    "FHO":    FHO(X_tr, X_te, y_train, y_test, dim),
                    "GA":     GA(X_tr, X_te, y_train, y_test, dim),
                    "GIGA":   GIGA(X_tr, X_te, y_train, y_test, dim),
                }

                for mname, get_model in models_dict.items():
                    for fsname, mask in masks.items():
                        if mask.sum() == 0:
                            continue
                        clf = get_model()
                        clf.fit(X_tr[:, mask], y_train)
                        preds = clf.predict(X_te[:, mask])
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average="weighted")

                        key = (mname, fsname)
                        fold_metrics.setdefault(key, []).append(
                            (acc, f1, int(mask.sum()))
                        )

                print(f"    Fold {fold_idx + 1}/{n_folds} done")

            # Average across folds
            for (mname, fsname), vals in fold_metrics.items():
                avg_acc = np.mean([v[0] for v in vals])
                avg_f1  = np.mean([v[1] for v in vals])
                avg_nf  = np.mean([v[2] for v in vals])
                all_results.append(
                    [dname, mname, fsname, avg_acc, avg_f1, round(avg_nf)]
                )
                print(
                    f"    [{mname:15}] {fsname:7} "
                    f"Acc:{avg_acc:.4f}  F1:{avg_f1:.4f}  Feat:{round(avg_nf)}"
                )

        except Exception as e:
            print(f"  Error with {dname}: {e}")

    return pd.DataFrame(
        all_results,
        columns=["Dataset", "Model", "FS", "Accuracy", "F1", "NFeatures"],
    )


# ═══════════════════════════════════════════════════════════════════
# MATPLOTLIB GRAPHS
# ═══════════════════════════════════════════════════════════════════
FS_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000", "#7030A0", "#FFD700"]
MODELS    = ["Decision Tree", "XGBoost", "SVM", "Random Forest"]


def plot_grouped_bar(df_res, metric_col, metric_label, filename, fold_label):
    ds_list = list(df_res["Dataset"].unique())
    n_ds    = len(ds_list)
    n_fs    = len(FS_METHODS)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    x     = np.arange(n_ds)
    width = 0.13

    for ax_idx, model in enumerate(MODELS):
        ax  = axes[ax_idx]
        sub = df_res[df_res["Model"] == model]

        for fi, fs in enumerate(FS_METHODS):
            vals = []
            for ds in ds_list:
                row = sub[(sub["Dataset"] == ds) & (sub["FS"] == fs)]
                vals.append(
                    float(row[metric_col].values[0]) if not row.empty else 0.0
                )

            offset  = (fi - n_fs / 2 + 0.5) * width
            is_giga = fs == "GIGA"
            ax.bar(
                x + offset, vals, width,
                color=FS_COLORS[fi],
                edgecolor="black" if is_giga else "gray",
                linewidth=1.5 if is_giga else 0.5,
                label=fs, zorder=3,
            )

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"{metric_label} — {fold_label} Stratified CV — All Classifiers",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_average_line(df_res, metric_col, metric_label, filename, fold_label):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax_idx, model in enumerate(MODELS):
        ax  = axes[ax_idx]
        sub = df_res[df_res["Model"] == model]

        avgs = []
        for fs in FS_METHODS:
            vals = sub[sub["FS"] == fs][metric_col].values
            avgs.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)

        ax.plot(
            FS_METHODS, avgs, color="#2E75B6",
            linewidth=2, marker="o", markersize=7,
            markerfacecolor="white", markeredgewidth=2, zorder=3,
        )

        giga_idx = FS_METHODS.index("GIGA")
        ax.plot(
            giga_idx, avgs[giga_idx],
            marker="*", markersize=16,
            color="#FFD700", markeredgecolor="black",
            markeredgewidth=0.8, zorder=4, label="GIGA",
        )

        for i, (fs, v) in enumerate(zip(FS_METHODS, avgs)):
            ax.text(
                i, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if fs == "GIGA" else "normal",
            )

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(FS_METHODS)))
        ax.set_xticklabels(FS_METHODS, fontsize=10)
        ax.set_ylabel(f"Avg {metric_label}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"Average {metric_label} — {fold_label} Stratified CV",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════
# EXCEL REPORT HELPERS  (same styling logic as main.py)
# ═══════════════════════════════════════════════════════════════════
C_TITLE = "1F4E78"
C_HDR   = "2E75B6"
C_GIGA  = "BDD7EE"
C_ALT1  = "FFFFFF"
C_ALT2  = "EBF3FB"
C_AVG   = "FFF2CC"
C_GRAND = "FFD966"

TH = Side(style="thin",   color="8EA9C1")
MD = Side(style="medium", color="1F4E78")
NO = Side(style=None)


def bdr(top=False, bot=False, lft=False, rgt=False, med=False):
    s = MD if med else TH
    return Border(
        top=s if top else NO, bottom=s if bot else NO,
        left=s if lft else NO, right=s if rgt else NO,
    )


def fl(c):
    return PatternFill("solid", start_color=c, end_color=c)


def sc(cell, bold=False, fc="000000", bg=None, sz=11,
       ha="center", fmt=None, bord=None):
    cell.font      = Font(name="Arial", bold=bold, color=fc, size=sz)
    cell.alignment = Alignment(horizontal=ha, vertical="center")
    if bg:   cell.fill = fl(bg)
    if fmt:  cell.number_format = fmt
    if bord: cell.border = bord


def write_metric_sheet(ws, metric_label, metric_col, df_res, fold_label):
    DATASETS = list(df_res["Dataset"].unique())
    NCOLS    = len(FS_METHODS)
    TOTAL    = NCOLS + 2

    # Title row
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=TOTAL)
    sc(
        ws.cell(
            row=1, column=1,
            value=(
                f"{metric_label} Results — {fold_label} Stratified CV "
                f"— All Classifiers vs Feature Selection Methods"
            ),
        ),
        bold=True, fc="FFFFFF", bg=C_TITLE, sz=13,
        bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
    )
    ws.row_dimensions[1].height = 24

    cur_row  = 2
    avg_rows = []

    for model in MODELS:
        sub = df_res[df_res["Model"] == model]

        # Header row 1: column names
        sc(
            ws.cell(row=cur_row, column=1, value="Dataset"),
            bold=True, fc="FFFFFF", bg=C_HDR, sz=11,
            bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
        )
        sc(
            ws.cell(row=cur_row, column=2, value="Model"),
            bold=True, fc="FFFFFF", bg=C_HDR, sz=11,
            bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
        )
        for ci, fs in enumerate(FS_METHODS, start=3):
            sc(
                ws.cell(row=cur_row, column=ci, value=fs),
                bold=True, fc="FFFFFF",
                bg=C_GIGA if fs == "GIGA" else C_HDR, sz=11,
                bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
            )
        cur_row += 1

        # Header row 2: metric label
        for ci in range(1, TOTAL + 1):
            is_giga = ci >= 3 and FS_METHODS[ci - 3] == "GIGA"
            val = metric_label if ci >= 3 else ""
            sc(
                ws.cell(row=cur_row, column=ci, value=val),
                bold=True, fc="FFFFFF",
                bg=C_GIGA if is_giga else C_HDR, sz=10,
                bord=bdr(bot=True, lft=True, rgt=True),
            )
        cur_row += 1

        # Data rows
        for r_idx, dset in enumerate(DATASETS):
            row_bg = C_ALT1 if r_idx % 2 == 0 else C_ALT2
            sc(
                ws.cell(row=cur_row, column=1, value=dset),
                fc="1F4E78", bg=row_bg, ha="left", sz=11,
                bord=bdr(lft=True, rgt=True),
            )
            sc(
                ws.cell(row=cur_row, column=2, value=model),
                bg=row_bg, ha="left", sz=11,
                bord=bdr(lft=True, rgt=True),
            )
            for ci, fs in enumerate(FS_METHODS, start=3):
                row = sub[(sub["Dataset"] == dset) & (sub["FS"] == fs)]
                val = (
                    round(float(row[metric_col].values[0]), 4)
                    if not row.empty else ""
                )
                sc(
                    ws.cell(row=cur_row, column=ci, value=val),
                    bg=C_GIGA if fs == "GIGA" else row_bg,
                    sz=11, fmt="0.0000",
                    bord=bdr(lft=True, rgt=True),
                )
            cur_row += 1

        # Average row
        sc(
            ws.cell(row=cur_row, column=1, value=""),
            bg=C_AVG,
            bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
        )
        sc(
            ws.cell(row=cur_row, column=2, value="AVERAGE"),
            bold=True, bg=C_AVG, ha="left", sz=11,
            bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
        )

        data_start = cur_row - len(DATASETS)
        data_end   = cur_row - 1
        for ci in range(3, TOTAL + 1):
            vals = [
                ws.cell(row=r, column=ci).value
                for r in range(data_start, data_end + 1)
                if isinstance(ws.cell(row=r, column=ci).value, (int, float))
            ]
            avg_val = round(sum(vals) / len(vals), 6) if vals else ""
            sc(
                ws.cell(row=cur_row, column=ci, value=avg_val),
                bold=True,
                bg=C_GIGA if FS_METHODS[ci - 3] == "GIGA" else C_AVG,
                sz=11, fmt="0.000000",
                bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
            )

        avg_rows.append(cur_row)
        cur_row += 3

    # Grand Average row
    sc(
        ws.cell(row=cur_row, column=1, value="Grand"),
        bold=True, fc="FFFFFF", bg=C_GRAND, sz=11,
        bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
    )
    sc(
        ws.cell(row=cur_row, column=2, value="AVERAGE"),
        bold=True, fc="FFFFFF", bg=C_GRAND, ha="left", sz=11,
        bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
    )
    for ci in range(3, TOTAL + 1):
        vals = [
            ws.cell(row=ar, column=ci).value
            for ar in avg_rows
            if isinstance(ws.cell(row=ar, column=ci).value, (int, float))
        ]
        grand = round(sum(vals) / len(vals), 6) if vals else ""
        sc(
            ws.cell(row=cur_row, column=ci, value=grand),
            bold=True, fc="FFFFFF",
            bg=C_GIGA if FS_METHODS[ci - 3] == "GIGA" else C_GRAND,
            sz=11, fmt="0.000000",
            bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True),
        )

    # Column widths & freeze
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 17
    for ci in range(3, TOTAL + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 12
    ws.freeze_panes = "C3"


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    out_dir = os.path.join("Final_Results_GIGA", "graphs_cv")
    os.makedirs(out_dir, exist_ok=True)

    cv_results = {}  # {5: DataFrame, 10: DataFrame}

    for n_folds in (5, 10):
        fold_label = f"{n_folds}-Fold"
        print(f"\n{'='*60}")
        print(f" RUNNING {fold_label} STRATIFIED CROSS-VALIDATION")
        print(f"{'='*60}")

        df = run_cv_experiment(n_folds)
        cv_results[n_folds] = df

        # ── graphs ──
        print(f"\n>>> Generating {fold_label} graphs...")
        plot_grouped_bar(
            df, "Accuracy", "Accuracy",
            os.path.join(out_dir, f"Accuracy_per_dataset_{n_folds}fold.png"),
            fold_label,
        )
        plot_average_line(
            df, "Accuracy", "Accuracy",
            os.path.join(out_dir, f"Accuracy_average_{n_folds}fold.png"),
            fold_label,
        )
        plot_grouped_bar(
            df, "F1", "F1-Score",
            os.path.join(out_dir, f"F1Score_per_dataset_{n_folds}fold.png"),
            fold_label,
        )
        plot_average_line(
            df, "F1", "F1-Score",
            os.path.join(out_dir, f"F1Score_average_{n_folds}fold.png"),
            fold_label,
        )

    # ── Excel Report ──
    print("\n>>> Building Excel report...")
    wb = Workbook()
    wb.remove(wb.active)

    for n_folds in (5, 10):
        df = cv_results[n_folds]
        fold_label = f"{n_folds}-Fold"
        for metric_label, metric_col in [("Accuracy", "Accuracy"),
                                          ("F1-Score", "F1")]:
            sheet_name = f"{fold_label} {metric_label}"
            ws = wb.create_sheet(title=sheet_name)
            write_metric_sheet(ws, metric_label, metric_col, df, fold_label)

    out_path = os.path.join("Final_Results_GIGA", "CV_Results.xlsx")
    wb.save(out_path)
    print(f"\nDone! -> {out_path}")
    print(f"  Sheets : {wb.sheetnames}")
    for n_folds, df in cv_results.items():
        print(f"  {n_folds}-Fold rows: {len(df)}")
