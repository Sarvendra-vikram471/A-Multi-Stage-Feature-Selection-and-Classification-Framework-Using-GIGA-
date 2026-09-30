"""
Runtime Benchmark Script
========================
Measures wall-clock execution time (seconds) for each feature-selection
algorithm (GWO, WHO, FHO, GA, GIGA) across all datasets using the
same 70/30 train-test split as main.py.

Outputs:
  - Console table
  - runtime_comparison.csv
  - runtime_comparison.xlsx  (styled)
"""

import numpy as np
import pandas as pd
import time
import warnings
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────
# Fitness + helpers  (copied from main.py)
# ─────────────────────────────────────────────────────────────────
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


# ─── GWO ──────────────────────────────────────────────────────────
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


# ─── WHO ──────────────────────────────────────────────────────────
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


# ─── FHO ──────────────────────────────────────────────────────────
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


# ─── GA ───────────────────────────────────────────────────────────
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


# ─── GIGA ─────────────────────────────────────────────────────────
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

    # Stage 3: post-refinement
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


# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
def load_and_preprocess(path):
    df = pd.read_csv(path)
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.median()).values
    return X, y


RESULTS_DIR = "Final_Results_GIGA"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

FS_ALGORITHMS = {
    "GWO":  GWO,
    "WHO":  WHO,
    "FHO":  FHO,
    "GA":   GA,
    "GIGA": GIGA,
}

N_RUNS = 3  # average over multiple runs for stability


# ─────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = []  # (dataset, n_samples, n_features, algo, time_sec)

    for dname, path in datasets.items():
        if not os.path.exists(path):
            print(f"Skipping {dname} -- file not found")
            continue

        X_raw, y = load_and_preprocess(path)
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.3, random_state=42
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        dim = X_tr.shape[1]

        print(f"\n{dname}  (N={X_raw.shape[0]}, D={dim})")

        for algo_name, algo_fn in FS_ALGORITHMS.items():
            times = []
            for run in range(N_RUNS):
                np.random.seed(42 + run)
                t0 = time.perf_counter()
                algo_fn(X_tr, X_te, y_train, y_test, dim)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            avg_t = np.mean(times)
            std_t = np.std(times)
            rows.append([dname, X_raw.shape[0], dim, algo_name, avg_t, std_t])
            print(f"  {algo_name:6s}  {avg_t:8.4f}s  (+/- {std_t:.4f}s)")

    # ── Build DataFrame ──
    df = pd.DataFrame(
        rows,
        columns=["Dataset", "N_Samples", "N_Features", "Algorithm",
                 "Time_sec", "Std_sec"],
    )

    # Pivot for the table
    pivot = df.pivot_table(
        index="Dataset", columns="Algorithm", values="Time_sec"
    )[["GWO", "WHO", "FHO", "GA", "GIGA"]]

    # Add average row
    avg_row = pivot.mean()
    pivot.loc["AVERAGE"] = avg_row

    print("\n" + "=" * 65)
    print("  RUNTIME COMPARISON TABLE  (seconds, averaged over "
          f"{N_RUNS} runs)")
    print("=" * 65)
    print(pivot.to_string(float_format="%.4f"))

    # ── Save CSV ──
    csv_path = os.path.join(RESULTS_DIR, "runtime_comparison.csv")
    pivot.to_csv(csv_path, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    # ── Save styled Excel ──
    wb = Workbook()
    ws = wb.active
    ws.title = "Runtime (seconds)"

    # Styling constants
    C_TITLE = "1F4E78"
    C_HDR   = "2E75B6"
    C_GIGA  = "BDD7EE"
    C_AVG   = "FFF2CC"
    C_ALT1  = "FFFFFF"
    C_ALT2  = "EBF3FB"
    TH = Side(style="thin", color="8EA9C1")
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

    def scell(cell, bold=False, fc="000000", bg=None, sz=11,
              ha="center", fmt=None, bord=None):
        cell.font = Font(name="Arial", bold=bold, color=fc, size=sz)
        cell.alignment = Alignment(horizontal=ha, vertical="center")
        if bg:   cell.fill = fl(bg)
        if fmt:  cell.number_format = fmt
        if bord: cell.border = bord

    algos = ["GWO", "WHO", "FHO", "GA", "GIGA"]
    ds_list = [d for d in datasets if d in pivot.index]

    # Title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
    scell(ws.cell(row=1, column=1,
          value="Feature Selection Runtime Comparison (seconds)"),
          bold=True, fc="FFFFFF", bg=C_TITLE, sz=13,
          bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    # Header
    scell(ws.cell(row=2, column=1, value="Dataset"),
          bold=True, fc="FFFFFF", bg=C_HDR,
          bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    for ci, algo in enumerate(algos, start=2):
        scell(ws.cell(row=2, column=ci, value=algo),
              bold=True, fc="FFFFFF",
              bg=C_GIGA if algo == "GIGA" else C_HDR,
              bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    # Data rows
    for ri, ds in enumerate(ds_list):
        row_bg = C_ALT1 if ri % 2 == 0 else C_ALT2
        r = ri + 3
        scell(ws.cell(row=r, column=1, value=ds),
              fc="1F4E78", bg=row_bg, ha="left",
              bord=bdr(lft=True, rgt=True))
        for ci, algo in enumerate(algos, start=2):
            val = round(pivot.loc[ds, algo], 4)
            scell(ws.cell(row=r, column=ci, value=val),
                  bg=C_GIGA if algo == "GIGA" else row_bg,
                  fmt="0.0000",
                  bord=bdr(lft=True, rgt=True))

    # Average row
    r = len(ds_list) + 3
    scell(ws.cell(row=r, column=1, value="AVERAGE"),
          bold=True, bg=C_AVG, ha="left",
          bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    for ci, algo in enumerate(algos, start=2):
        val = round(pivot.loc["AVERAGE", algo], 4)
        scell(ws.cell(row=r, column=ci, value=val),
              bold=True,
              bg=C_GIGA if algo == "GIGA" else C_AVG,
              fmt="0.0000",
              bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    # Column widths
    ws.column_dimensions["A"].width = 14
    for ci in range(2, 7):
        ws.column_dimensions[get_column_letter(ci)].width = 12
    ws.freeze_panes = "B3"

    xlsx_path = os.path.join(RESULTS_DIR, "runtime_comparison.xlsx")
    wb.save(xlsx_path)
    print(f"Saved: {xlsx_path}")
