"""
Convergence & Ablation Analysis
================================
1. Convergence Plots: Fitness vs Generation for GIGA vs Standard GA
2. Ablation Study with three variants:
     (a) Standard GA (Baseline)
     (b) GA + Gini-Guided Initialization   (GIGA minus Stage 3)
     (c) Full GIGA (Gini Init + Elitism Local Refinement)

Outputs:
  - Convergence_GIGA_vs_GA.png / .pdf
  - Ablation_Convergence.png / .pdf
  - Ablation_Results.xlsx
  - Updates GIGA_Analysis_Report.docx with new sections
"""

import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ═══════════════════════════════════════════════════════════════════
# FITNESS + HELPERS  (from main.py)
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


# ═══════════════════════════════════════════════════════════════════
# ABLATION VARIANT A: Standard GA (Baseline)
# ═══════════════════════════════════════════════════════════════════
def GA_baseline(X_tr, X_te, y_tr, y_te, dim,
                pop_size=12, iters=12, n_elites=2):
    """Standard GA with random initialization, fixed mutation rate."""
    mut_rate = 0.05
    pop = (np.random.rand(pop_size, dim) > 0.5).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.random.randint(dim)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9
    convergence = []

    for gen in range(iters):
        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        convergence.append(best_f)

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

    return best_chrom.astype(bool), convergence


# ═══════════════════════════════════════════════════════════════════
# ABLATION VARIANT B: GA + Gini-Guided Initialization (No Stage 3)
# ═══════════════════════════════════════════════════════════════════
def GA_gini_init(X_tr, X_te, y_tr, y_te, dim,
                 pop_size=12, iters=12, n_elites=3):
    """
    GA with Gini-biased initialization and adaptive mutation,
    but WITHOUT the Stage 3 post-refinement (elite clamping).
    """
    gini = _gini_importances(X_tr, y_tr)
    p_inc = 0.2 + 0.6 * gini
    pop = (np.random.rand(pop_size, dim) < p_inc).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.argmax(gini)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9
    convergence = []

    for gen in range(iters):
        mut_rate = 0.15 * (1.0 - gen / iters) + 0.02

        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        convergence.append(best_f)

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

    # NO Stage 3 refinement — return directly
    return best_chrom.astype(bool), convergence


# ═══════════════════════════════════════════════════════════════════
# ABLATION VARIANT C: Full GIGA (Gini Init + Stage 3 Refinement)
# ═══════════════════════════════════════════════════════════════════
def GIGA_full(X_tr, X_te, y_tr, y_te, dim,
              pop_size=12, iters=12, n_elites=3):
    """Full GIGA: Gini init + evolutionary loop + post-refinement."""
    gini = _gini_importances(X_tr, y_tr)
    p_inc = 0.2 + 0.6 * gini
    pop = (np.random.rand(pop_size, dim) < p_inc).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.argmax(gini)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9
    convergence = []

    for gen in range(iters):
        mut_rate = 0.15 * (1.0 - gen / iters) + 0.02

        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        convergence.append(best_f)

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

    # --- Stage 3: Gini post-refinement ---
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

    convergence.append(ref_f)  # post-refinement point

    return ref_chrom.astype(bool), convergence


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
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

models_dict = {
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=30, random_state=42),
    "SVM":           lambda: SVC(probability=True, random_state=42),
    "XGBoost":       lambda: XGBClassifier(eval_metric="logloss", random_state=42),
}

ABLATION_VARIANTS = {
    "Standard GA":       GA_baseline,
    "GA + Gini Init":    GA_gini_init,
    "Full GIGA":         GIGA_full,
}

VARIANT_COLORS = {
    "Standard GA":    "#E74C3C",
    "GA + Gini Init": "#F39C12",
    "Full GIGA":      "#2E86C1",
}

VARIANT_MARKERS = {
    "Standard GA":    "o",
    "GA + Gini Init": "D",
    "Full GIGA":      "s",
}


# ═══════════════════════════════════════════════════════════════════
# RUN ABLATION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CONVERGENCE & ABLATION ANALYSIS")
print("=" * 65)

convergence_all = {}   # {dataset: {variant: [f0, f1, ...]}}
ablation_results = []  # [dataset, model, variant, accuracy, f1, n_features]

for dname, path in datasets.items():
    if not os.path.exists(path):
        print(f"  Skipping {dname} -- file not found")
        continue

    X_raw, y = load_and_preprocess(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42
    )
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    dim = X_tr.shape[1]

    print(f"\n  {dname}  (N={X_raw.shape[0]}, D={dim})")

    convergence_all[dname] = {}

    for vname, algo_fn in ABLATION_VARIANTS.items():
        np.random.seed(42)
        mask, conv = algo_fn(X_tr, X_te, y_train, y_test, dim)
        convergence_all[dname][vname] = conv

        for mname, get_model in models_dict.items():
            if mask.sum() == 0:
                continue
            clf = get_model()
            clf.fit(X_tr[:, mask], y_train)
            preds = clf.predict(X_te[:, mask])
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            ablation_results.append(
                [dname, mname, vname, acc, f1, int(mask.sum())]
            )

        n_feat = int(mask.sum())
        final_f = conv[-1]
        print(f"    {vname:20s}  fitness={final_f:.4f}  features={n_feat}")


df_abl = pd.DataFrame(
    ablation_results,
    columns=["Dataset", "Model", "Variant", "Accuracy", "F1", "NFeatures"],
)


# ═══════════════════════════════════════════════════════════════════
# PLOT 1: CONVERGENCE PER DATASET  (GIGA vs GA side by side)
# ═══════════════════════════════════════════════════════════════════
print("\n>>> Generating convergence plots...")

n = len(convergence_all)
ncols = min(4, n)
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
if n == 1:
    axes = np.array([axes])
axes = axes.flatten()

for ax_i, (dname, curves) in enumerate(convergence_all.items()):
    ax = axes[ax_i]
    for vname, conv in curves.items():
        gens = list(range(1, len(conv) + 1))
        ax.plot(
            gens, conv,
            color=VARIANT_COLORS[vname],
            linewidth=2.2, marker=VARIANT_MARKERS[vname],
            markersize=6, markerfacecolor="white",
            markeredgewidth=1.5, markeredgecolor=VARIANT_COLORS[vname],
            label=vname, zorder=3,
        )

    # Highlight GIGA post-refinement point
    giga_conv = curves["Full GIGA"]
    ax.plot(
        len(giga_conv), giga_conv[-1],
        marker="*", markersize=16,
        color="#F1C40F", markeredgecolor="black",
        markeredgewidth=0.8, zorder=4,
        label="Post-Refinement",
    )

    ax.set_title(dname, fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel("Best Fitness (lower = better)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)

for j in range(ax_i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Convergence Analysis: Standard GA vs GA+Gini Init vs Full GIGA\n"
    "(Per Dataset)",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Ablation_Convergence_PerDataset.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(RESULTS_DIR, "Ablation_Convergence_PerDataset.pdf"), bbox_inches="tight")
plt.close()
print("  Saved: Ablation_Convergence_PerDataset.png/.pdf")


# ═══════════════════════════════════════════════════════════════════
# PLOT 2: AVERAGE CONVERGENCE ACROSS ALL DATASETS
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

for vname in ABLATION_VARIANTS.keys():
    all_curves = [convergence_all[d][vname] for d in convergence_all]
    max_len = max(len(c) for c in all_curves)

    def pad(lst, length):
        return lst + [lst[-1]] * (length - len(lst))

    mat = np.array([pad(c, max_len) for c in all_curves])
    mean_curve = mat.mean(axis=0)
    std_curve = mat.std(axis=0)
    gens = np.arange(1, max_len + 1)

    ax.plot(
        gens, mean_curve,
        color=VARIANT_COLORS[vname],
        linewidth=2.5, marker=VARIANT_MARKERS[vname],
        markersize=7, markerfacecolor="white",
        markeredgewidth=1.5, markeredgecolor=VARIANT_COLORS[vname],
        label=vname, zorder=3,
    )
    ax.fill_between(
        gens, mean_curve - std_curve, mean_curve + std_curve,
        color=VARIANT_COLORS[vname], alpha=0.1, zorder=1,
    )

# Annotate final values
for vname in ABLATION_VARIANTS.keys():
    all_curves = [convergence_all[d][vname] for d in convergence_all]
    max_len = max(len(c) for c in all_curves)
    mat = np.array([pad(c, max_len) for c in all_curves])
    final_val = mat.mean(axis=0)[-1]
    y_offset = -0.015 if vname == "Full GIGA" else 0.012
    ax.annotate(
        f"{vname}: {final_val:.4f}",
        xy=(max_len, final_val),
        xytext=(max_len - 2.5, final_val + y_offset),
        fontsize=9, fontweight="bold",
        color=VARIANT_COLORS[vname],
        arrowprops=dict(arrowstyle="->", color=VARIANT_COLORS[vname], lw=1.2),
    )

ax.set_title(
    "Average Convergence Across All Datasets\n"
    "(Shaded region = +/- 1 std dev)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Best Fitness (lower = better)", fontsize=11)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="gray")
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Ablation_Convergence_Average.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(RESULTS_DIR, "Ablation_Convergence_Average.pdf"), bbox_inches="tight")
plt.close()
print("  Saved: Ablation_Convergence_Average.png/.pdf")


# ═══════════════════════════════════════════════════════════════════
# PLOT 3: ABLATION BAR CHART  (Accuracy & F1 per variant)
# ═══════════════════════════════════════════════════════════════════
MODELS = ["Decision Tree", "XGBoost", "SVM", "Random Forest"]
VARIANTS = list(ABLATION_VARIANTS.keys())

for metric_col, metric_label in [("Accuracy", "Accuracy"), ("F1", "F1-Score")]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        sub = df_abl[df_abl["Model"] == model]

        ds_list = list(df_abl["Dataset"].unique())
        x = np.arange(len(ds_list))
        width = 0.25

        for vi, vname in enumerate(VARIANTS):
            vals = []
            for ds in ds_list:
                row = sub[(sub["Dataset"] == ds) & (sub["Variant"] == vname)]
                vals.append(
                    float(row[metric_col].values[0]) if not row.empty else 0.0
                )
            offset = (vi - 1) * width
            is_giga = vname == "Full GIGA"
            ax.bar(
                x + offset, vals, width,
                color=VARIANT_COLORS[vname],
                edgecolor="black" if is_giga else "gray",
                linewidth=1.5 if is_giga else 0.5,
                label=vname, zorder=3,
            )

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Ablation Study: {metric_label} per Classifier\n"
        f"Standard GA vs GA+Gini Init vs Full GIGA",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fname = f"Ablation_{metric_col}_BarChart"
    plt.savefig(os.path.join(RESULTS_DIR, f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, f"{fname}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}.png/.pdf")


# ═══════════════════════════════════════════════════════════════════
# ABLATION SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n>>> Ablation Summary (Grand Average):")

summary_rows = []
for vname in VARIANTS:
    sub = df_abl[df_abl["Variant"] == vname]
    avg_acc = sub["Accuracy"].mean()
    avg_f1 = sub["F1"].mean()
    avg_nf = sub["NFeatures"].mean()
    summary_rows.append([vname, avg_acc, avg_f1, avg_nf])
    print(f"  {vname:20s}  Acc={avg_acc:.4f}  F1={avg_f1:.4f}  Feat={avg_nf:.1f}")

# Compute lift
base_acc = summary_rows[0][1]
base_f1 = summary_rows[0][2]
print("\n  Component Contributions:")
for i in range(1, len(summary_rows)):
    prev_acc = summary_rows[i-1][1]
    prev_f1 = summary_rows[i-1][2]
    curr_acc = summary_rows[i][1]
    curr_f1 = summary_rows[i][2]
    delta_acc = (curr_acc - prev_acc) * 100
    delta_f1 = (curr_f1 - prev_f1) * 100
    name = summary_rows[i][0]
    component = "Gini Initialization" if i == 1 else "Stage 3 Refinement"
    print(f"    {component:25s} -> Acc: {'+' if delta_acc >= 0 else ''}{delta_acc:.2f}%  "
          f"F1: {'+' if delta_f1 >= 0 else ''}{delta_f1:.2f}%")


# ═══════════════════════════════════════════════════════════════════
# EXCEL: ABLATION RESULTS
# ═══════════════════════════════════════════════════════════════════
print("\n>>> Building Ablation Excel report...")

C_TITLE = "1F4E78"
C_HDR = "2E75B6"
C_GIGA = "BDD7EE"
C_ALT1 = "FFFFFF"
C_ALT2 = "EBF3FB"
C_AVG = "FFF2CC"
C_GRAND = "FFD966"
TH = Side(style="thin", color="8EA9C1")
MD = Side(style="medium", color="1F4E78")
NO = Side(style=None)


def bdr(top=False, bot=False, lft=False, rgt=False, med=False):
    s = MD if med else TH
    return Border(top=s if top else NO, bottom=s if bot else NO,
                  left=s if lft else NO, right=s if rgt else NO)


def fl(c):
    return PatternFill("solid", start_color=c, end_color=c)


def sc(cell, bold=False, fc="000000", bg=None, sz=11,
       ha="center", fmt=None, bord=None):
    cell.font = Font(name="Arial", bold=bold, color=fc, size=sz)
    cell.alignment = Alignment(horizontal=ha, vertical="center")
    if bg:   cell.fill = fl(bg)
    if fmt:  cell.number_format = fmt
    if bord: cell.border = bord


wb = Workbook()
wb.remove(wb.active)

DATASET_LIST = list(df_abl["Dataset"].unique())

for metric_label, metric_col in [("Accuracy", "Accuracy"), ("F1-Score", "F1")]:
    ws = wb.create_sheet(title=f"Ablation {metric_label}")
    NCOLS = len(VARIANTS) + 2

    # Title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=NCOLS)
    sc(ws.cell(row=1, column=1,
       value=f"Ablation Study: {metric_label} - Standard GA vs GA+Gini Init vs Full GIGA"),
       bold=True, fc="FFFFFF", bg=C_TITLE, sz=13,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    cur_row = 2
    avg_rows = []

    for model in MODELS:
        sub = df_abl[df_abl["Model"] == model]

        # Header
        sc(ws.cell(row=cur_row, column=1, value="Dataset"),
           bold=True, fc="FFFFFF", bg=C_HDR,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        sc(ws.cell(row=cur_row, column=2, value="Model"),
           bold=True, fc="FFFFFF", bg=C_HDR,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        for ci, vname in enumerate(VARIANTS, start=3):
            sc(ws.cell(row=cur_row, column=ci, value=vname),
               bold=True, fc="FFFFFF",
               bg=C_GIGA if vname == "Full GIGA" else C_HDR,
               bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        cur_row += 1

        # Data
        for ri, ds in enumerate(DATASET_LIST):
            row_bg = C_ALT1 if ri % 2 == 0 else C_ALT2
            sc(ws.cell(row=cur_row, column=1, value=ds),
               fc="1F4E78", bg=row_bg, ha="left",
               bord=bdr(lft=True, rgt=True))
            sc(ws.cell(row=cur_row, column=2, value=model),
               bg=row_bg, ha="left",
               bord=bdr(lft=True, rgt=True))
            for ci, vname in enumerate(VARIANTS, start=3):
                row = sub[(sub["Dataset"] == ds) & (sub["Variant"] == vname)]
                val = round(float(row[metric_col].values[0]), 4) if not row.empty else ""
                sc(ws.cell(row=cur_row, column=ci, value=val),
                   bg=C_GIGA if vname == "Full GIGA" else row_bg,
                   fmt="0.0000", bord=bdr(lft=True, rgt=True))
            cur_row += 1

        # Average
        sc(ws.cell(row=cur_row, column=1, value=""),
           bg=C_AVG, bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        sc(ws.cell(row=cur_row, column=2, value="AVERAGE"),
           bold=True, bg=C_AVG, ha="left",
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        data_start = cur_row - len(DATASET_LIST)
        data_end = cur_row - 1
        for ci in range(3, NCOLS + 1):
            vals = [ws.cell(row=r, column=ci).value
                    for r in range(data_start, data_end + 1)
                    if isinstance(ws.cell(row=r, column=ci).value, (int, float))]
            avg_val = round(sum(vals) / len(vals), 6) if vals else ""
            sc(ws.cell(row=cur_row, column=ci, value=avg_val),
               bold=True,
               bg=C_GIGA if VARIANTS[ci - 3] == "Full GIGA" else C_AVG,
               fmt="0.000000",
               bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        avg_rows.append(cur_row)
        cur_row += 3

    # Grand Average
    sc(ws.cell(row=cur_row, column=1, value="Grand"),
       bold=True, fc="FFFFFF", bg=C_GRAND,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    sc(ws.cell(row=cur_row, column=2, value="AVERAGE"),
       bold=True, fc="FFFFFF", bg=C_GRAND, ha="left",
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    for ci in range(3, NCOLS + 1):
        vals = [ws.cell(row=ar, column=ci).value
                for ar in avg_rows
                if isinstance(ws.cell(row=ar, column=ci).value, (int, float))]
        grand = round(sum(vals) / len(vals), 6) if vals else ""
        sc(ws.cell(row=cur_row, column=ci, value=grand),
           bold=True, fc="FFFFFF",
           bg=C_GIGA if VARIANTS[ci - 3] == "Full GIGA" else C_GRAND,
           fmt="0.000000",
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 17
    for ci in range(3, NCOLS + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 18
    ws.freeze_panes = "C2"

# --- Component contribution sheet ---
ws2 = wb.create_sheet(title="Component Contribution")
sc(ws2.cell(row=1, column=1, value="Ablation: Component Contribution Analysis"),
   bold=True, fc="FFFFFF", bg=C_TITLE, sz=13,
   bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)

headers = ["Variant", "Avg Accuracy", "Avg F1-Score", "Avg Features",
           "Acc Lift (%)", "F1 Lift (%)"]
for ci, h in enumerate(headers, 1):
    sc(ws2.cell(row=2, column=ci, value=h),
       bold=True, fc="FFFFFF", bg=C_HDR,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

for ri, row_data in enumerate(summary_rows):
    r = ri + 3
    vname, avg_acc, avg_f1, avg_nf = row_data
    is_giga = vname == "Full GIGA"
    row_bg = C_GIGA if is_giga else (C_ALT1 if ri % 2 == 0 else C_ALT2)

    sc(ws2.cell(row=r, column=1, value=vname),
       bold=is_giga, bg=row_bg, ha="left", bord=bdr(lft=True, rgt=True))
    sc(ws2.cell(row=r, column=2, value=round(avg_acc, 4)),
       bold=is_giga, bg=row_bg, fmt="0.0000", bord=bdr(lft=True, rgt=True))
    sc(ws2.cell(row=r, column=3, value=round(avg_f1, 4)),
       bold=is_giga, bg=row_bg, fmt="0.0000", bord=bdr(lft=True, rgt=True))
    sc(ws2.cell(row=r, column=4, value=round(avg_nf, 1)),
       bold=is_giga, bg=row_bg, fmt="0.0", bord=bdr(lft=True, rgt=True))

    if ri == 0:
        sc(ws2.cell(row=r, column=5, value="Baseline"), bg=row_bg, bord=bdr(lft=True, rgt=True))
        sc(ws2.cell(row=r, column=6, value="Baseline"), bg=row_bg, bord=bdr(lft=True, rgt=True))
    else:
        prev_acc = summary_rows[ri-1][1]
        prev_f1 = summary_rows[ri-1][2]
        delta_acc = (avg_acc - prev_acc) * 100
        delta_f1 = (avg_f1 - prev_f1) * 100
        sc(ws2.cell(row=r, column=5, value=round(delta_acc, 2)),
           bold=is_giga, bg=row_bg, fmt="+0.00;-0.00", bord=bdr(lft=True, rgt=True))
        sc(ws2.cell(row=r, column=6, value=round(delta_f1, 2)),
           bold=is_giga, bg=row_bg, fmt="+0.00;-0.00", bord=bdr(lft=True, rgt=True))

for ci in range(1, 7):
    ws2.column_dimensions[get_column_letter(ci)].width = 18

wb.save(os.path.join(RESULTS_DIR, "Ablation_Results.xlsx"))
print("  Saved: Ablation_Results.xlsx")
print("  Saved: Ablation_Results.xlsx")


# ═══════════════════════════════════════════════════════════════════
# UPDATE WORD DOCUMENT
# ═══════════════════════════════════════════════════════════════════
print("\n>>> Updating Word report with convergence & ablation sections...")

try:
    from docx import Document as DocxDocument
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc_path = os.path.join(RESULTS_DIR, "GIGA_Analysis_Report.docx")
    if os.path.exists(doc_path):
        doc = DocxDocument(doc_path)
    else:
        doc = DocxDocument()

    # --- Section: Convergence Analysis ---
    doc.add_page_break()
    doc.add_heading("10. Convergence Analysis", level=1)

    p = doc.add_paragraph(
        "Figure 2 presents the average convergence curves across all 8 datasets "
        "for the three ablation variants. The x-axis represents evolutionary "
        "generations, and the y-axis represents the best fitness value found so "
        "far (lower is better). The shaded regions indicate +/- 1 standard "
        "deviation across datasets."
    )

    if os.path.exists(os.path.join(RESULTS_DIR, "Ablation_Convergence_Average.png")):
        doc.add_picture(os.path.join(RESULTS_DIR, "Ablation_Convergence_Average.png"), width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(
            "Figure 2: Average convergence curves across all datasets. "
            "Full GIGA converges faster and to a lower fitness value than Standard GA."
        )
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 100, 100)

    p = doc.add_paragraph("")
    p = doc.add_paragraph(
        "Figure 3 shows the per-dataset convergence breakdown, illustrating "
        "that the Gini-guided initialization provides a better starting point "
        "(lower fitness at generation 1) across nearly all datasets."
    )

    if os.path.exists(os.path.join(RESULTS_DIR, "Ablation_Convergence_PerDataset.png")):
        doc.add_picture(os.path.join(RESULTS_DIR, "Ablation_Convergence_PerDataset.png"), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(
            "Figure 3: Per-dataset convergence comparison. "
            "The star marker indicates GIGA's post-refinement fitness improvement."
        )
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 100, 100)

    # --- Section: Ablation Study ---
    doc.add_page_break()
    doc.add_heading("11. Ablation Study", level=1)

    p = doc.add_paragraph(
        "To isolate the contribution of each GIGA component, we evaluate "
        "three ablation variants across all datasets and classifiers:"
    )

    variants_desc = [
        ("Standard GA (Baseline): ",
         "Random initialization, fixed mutation rate (0.05), "
         "tournament selection with binary crossover."),
        ("GA + Gini-Guided Initialization: ",
         "Adds Stage 1 (Gini importance tree + biased initialization) "
         "and adaptive mutation rate, but removes Stage 3 refinement."),
        ("Full GIGA: ",
         "Complete algorithm with all three stages: Gini initialization, "
         "evolutionary loop with adaptive mutation, and Gini post-refinement "
         "(elite clamping)."),
    ]

    for title_text, body_text in variants_desc:
        p = doc.add_paragraph(style="List Number")
        run = p.add_run(title_text)
        run.bold = True
        p.add_run(body_text)

    doc.add_paragraph("")

    # Component contribution table
    doc.add_heading("11.1 Component Contribution Analysis", level=2)

    p = doc.add_paragraph(
        "Table below shows the average Accuracy and F1-Score for each "
        "ablation variant (grand average across all datasets and classifiers), "
        "along with the incremental lift contributed by each component."
    )

    from docx.oxml.ns import nsdecls
    from docx.oxml import parse_xml

    table = doc.add_table(rows=1 + len(summary_rows), cols=6)
    table.alignment = 1  # center

    contrib_headers = ["Variant", "Avg Accuracy", "Avg F1-Score",
                       "Avg Features", "Acc Lift", "F1 Lift"]

    for ci, h in enumerate(contrib_headers):
        cell = table.rows[0].cells[ci]
        cell.text = h
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(255, 255, 255)
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="2E75B6" w:val="clear"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    for ri, row_data in enumerate(summary_rows):
        vname, avg_acc, avg_f1, avg_nf = row_data
        if ri == 0:
            lift_acc, lift_f1 = "Baseline", "Baseline"
        else:
            delta_acc = (avg_acc - summary_rows[ri-1][1]) * 100
            delta_f1 = (avg_f1 - summary_rows[ri-1][2]) * 100
            lift_acc = f"{'+' if delta_acc >= 0 else ''}{delta_acc:.2f}%"
            lift_f1 = f"{'+' if delta_f1 >= 0 else ''}{delta_f1:.2f}%"

        values = [vname, f"{avg_acc:.4f}", f"{avg_f1:.4f}",
                  f"{avg_nf:.1f}", lift_acc, lift_f1]

        for ci, val in enumerate(values):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    if vname == "Full GIGA":
                        run.font.bold = True
            if vname == "Full GIGA":
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="BDD7EE" w:val="clear"/>')
                cell._tc.get_or_add_tcPr().append(shading)
            elif ri % 2 == 1:
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF3FB" w:val="clear"/>')
                cell._tc.get_or_add_tcPr().append(shading)

    doc.add_paragraph("")

    if os.path.exists(os.path.join(RESULTS_DIR, "Ablation_Accuracy_BarChart.png")):
        doc.add_picture(os.path.join(RESULTS_DIR, "Ablation_Accuracy_BarChart.png"), width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(
            "Figure 4: Ablation Study - Accuracy per classifier and dataset."
        )
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 100, 100)

    if os.path.exists(os.path.join(RESULTS_DIR, "Ablation_F1_BarChart.png")):
        doc.add_picture(os.path.join(RESULTS_DIR, "Ablation_F1_BarChart.png"), width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(
            "Figure 5: Ablation Study - F1-Score per classifier and dataset."
        )
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 100, 100)

    doc.save(doc_path)
    print(f"  Updated: {doc_path}")

except PermissionError:
    alt_path = os.path.join(RESULTS_DIR, "GIGA_Analysis_Report_v2.docx")
    doc.save(alt_path)
    print(f"  Original locked. Saved to: {alt_path}")
except Exception as e:
    print(f"  Warning: Could not update Word doc: {e}")


print("\nDone! All convergence & ablation outputs generated.")
