import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt

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

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SHARED FITNESS FUNCTION
# =============================================================================
def clamp(x):
    return np.clip(x, 0.0, 1.0)


def fitness(vec, X_train, X_test, y_train, y_test):
    mask = (vec > 0.5) if vec.dtype == float else vec.astype(bool)
    if mask.sum() == 0:
        return 2.0, mask
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train[:, mask], y_train)
    preds = clf.predict(X_test[:, mask])
    score = f1_score(y_test, preds, average='weighted')
    obj   = 0.9 * (1 - score) + 0.1 * (mask.sum() / X_train.shape[1])
    return obj, mask


# =============================================================================
# BASELINE METAHEURISTICS
# =============================================================================
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
            A      = a * (2 * np.random.rand() - 1)
            pop[i] = clamp(best_vec - A * np.abs(best_vec - pop[i]))
    return best_vec > 0.5


def WHO(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12):
    pop  = np.random.rand(pop_size, dim)
    best = pop[0].copy()
    for _ in range(iters):
        fb, _ = fitness(best, X_tr, X_te, y_tr, y_te)
        for i in range(pop_size):
            f, _ = fitness(pop[i], X_tr, X_te, y_tr, y_te)
            if f < fb:
                best, fb = pop[i].copy(), f
            step   = (best - pop[i]) if np.random.rand() > 0.5 \
                     else (pop[np.random.randint(pop_size)] - pop[i])
            pop[i] = clamp(pop[i] + np.random.rand() * step)
    return best > 0.5


def FHO(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12):
    pop  = np.random.rand(pop_size, dim)
    best = pop[0].copy()
    for _ in range(iters):
        fb, _ = fitness(best, X_tr, X_te, y_tr, y_te)
        for i in range(pop_size):
            f, _ = fitness(pop[i], X_tr, X_te, y_tr, y_te)
            if f < fb:
                best, fb = pop[i].copy(), f
            pop[i] = clamp(pop[i] + np.random.rand() * (best - pop[i]))
    return best > 0.5


# =============================================================================
# GA  (now returns convergence history)
# =============================================================================
def GA(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12, n_elites=2):
    mut_rate = 0.05
    pop = (np.random.rand(pop_size, dim) > 0.5).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.random.randint(dim)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9
    convergence = []                          # <<< NEW

    for gen in range(iters):
        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        convergence.append(best_f)            # <<< NEW

        elite_idx = np.argsort(fit_vals)[:n_elites]
        new_pop   = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[i1] if fit_vals[i1] < fit_vals[i2] else pop[i2]
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[i1] if fit_vals[i1] < fit_vals[i2] else pop[i2]
            pt    = np.random.randint(1, dim)
            child = np.concatenate([p1[:pt], p2[pt:]])
            flip        = np.random.rand(dim) < mut_rate
            child[flip] = 1 - child[flip]
            if child.sum() == 0:
                child[np.random.randint(dim)] = 1
            new_pop.append(child)

        pop = np.array(new_pop)

    return best_chrom.astype(bool), convergence   # <<< CHANGED


# =============================================================================
# GIGA  (now returns convergence history)
# =============================================================================
def _gini_importances(X_tr, y_tr):
    dt  = DecisionTreeClassifier(max_depth=5, random_state=42)
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
    flip        = np.random.rand(len(chrom)) < rate
    child       = chrom.copy()
    child[flip] = 1 - child[flip]
    return child


def GIGA(X_tr, X_te, y_tr, y_te, dim, pop_size=12, iters=12, n_elites=3):
    gini  = _gini_importances(X_tr, y_tr)
    p_inc = 0.2 + 0.6 * gini
    pop   = (np.random.rand(pop_size, dim) < p_inc).astype(int)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i][np.argmax(gini)] = 1

    best_chrom, best_f = pop[0].copy(), 1e9
    convergence = []                          # <<< NEW

    for gen in range(iters):
        mut_rate = 0.15 * (1.0 - gen / iters) + 0.02

        fit_vals = np.zeros(pop_size)
        for i in range(pop_size):
            f, _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)
            fit_vals[i] = f
            if f < best_f:
                best_f, best_chrom = f, pop[i].copy()

        convergence.append(best_f)            # <<< NEW

        elite_idx = np.argsort(fit_vals)[:n_elites]
        new_pop   = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < pop_size:
            child = _mutate(_crossover(
                        _tournament_select(pop, fit_vals),
                        _tournament_select(pop, fit_vals)), mut_rate)
            if child.sum() == 0:
                child[np.argmax(gini)] = 1
            new_pop.append(child)
        pop = np.array(new_pop)

    # --- Gini post-refinement (does NOT add extra generations) ---
    fit_vals = np.zeros(pop_size)
    for i in range(pop_size):
        fit_vals[i], _ = fitness(pop[i].astype(float), X_tr, X_te, y_tr, y_te)

    lo_thresh = np.median(gini)
    hi_thresh = np.percentile(gini, 75)
    ref_f     = best_f
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

    # record post-refinement fitness as a final point
    convergence.append(ref_f)                 # <<< NEW

    return ref_chrom.astype(bool), convergence  # <<< CHANGED


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_preprocess(path):
    df   = pd.read_csv(path)
    X_df = df.iloc[:, :-1]
    y    = df.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)
    X = X_df.apply(pd.to_numeric, errors='coerce').fillna(X_df.median()).values
    return X, y


# =============================================================================
# CONVERGENCE PLOT FUNCTIONS
# =============================================================================
def plot_convergence_per_dataset(conv_dict, out_dir="graphs"):
    """
    conv_dict: {dataset_name: {"GA": [f0,f1,...], "GIGA": [f0,f1,...,f_post]}}
    One subplot per dataset.
    """
    n = len(conv_dict)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_i, (dname, curves) in enumerate(conv_dict.items()):
        ax = axes[ax_i]
        ga_curve   = curves["GA"]
        giga_curve = curves["GIGA"]

        ga_gens   = list(range(1, len(ga_curve) + 1))
        giga_gens = list(range(1, len(giga_curve)))          # evolutionary gens
        giga_gens.append(len(giga_curve))                     # post-refinement

        ax.plot(ga_gens, ga_curve,
                color="#E74C3C", linewidth=2, marker="o", markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="Standard GA", zorder=3)
        ax.plot(giga_gens, giga_curve,
                color="#2E86C1", linewidth=2, marker="s", markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="GIGA (Gini-guided)", zorder=3)

        # highlight GIGA's final post-refinement point
        ax.plot(giga_gens[-1], giga_curve[-1],
                marker="*", markersize=14, color="#F1C40F",
                markeredgecolor="black", markeredgewidth=0.8,
                zorder=4, label="GIGA post-refine")

        ax.set_title(dname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Best Fitness (lower is better)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper right")

    # hide unused axes
    for j in range(ax_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergence: GIGA vs Standard GA — Per Dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "Convergence_per_dataset.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_convergence_average(conv_dict, out_dir="graphs"):
    """
    Grand average convergence curve across all datasets.
    Pads shorter histories with their last value so all are the same length.
    """
    if not conv_dict:
        return

    ga_all   = [c["GA"]   for c in conv_dict.values()]
    giga_all = [c["GIGA"] for c in conv_dict.values()]

    max_ga   = max(len(c) for c in ga_all)
    max_giga = max(len(c) for c in giga_all)

    def pad(lst, length):
        return lst + [lst[-1]] * (length - len(lst))

    ga_mat   = np.array([pad(c, max_ga)   for c in ga_all])
    giga_mat = np.array([pad(c, max_giga) for c in giga_all])

    ga_mean   = ga_mat.mean(axis=0)
    giga_mean = giga_mat.mean(axis=0)
    ga_std    = ga_mat.std(axis=0)
    giga_std  = giga_mat.std(axis=0)

    ga_gens   = np.arange(1, max_ga + 1)
    giga_gens = np.arange(1, max_giga + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # GA
    ax.plot(ga_gens, ga_mean, color="#E74C3C", linewidth=2.5,
            marker="o", markersize=6, markerfacecolor="white",
            markeredgewidth=1.5, label="Standard GA", zorder=3)
    ax.fill_between(ga_gens, ga_mean - ga_std, ga_mean + ga_std,
                    color="#E74C3C", alpha=0.12, zorder=1)

    # GIGA
    ax.plot(giga_gens, giga_mean, color="#2E86C1", linewidth=2.5,
            marker="s", markersize=6, markerfacecolor="white",
            markeredgewidth=1.5, label="GIGA (Gini-guided)", zorder=3)
    ax.fill_between(giga_gens, giga_mean - giga_std, giga_mean + giga_std,
                    color="#2E86C1", alpha=0.12, zorder=1)

    # highlight GIGA post-refinement
    ax.plot(giga_gens[-1], giga_mean[-1],
            marker="*", markersize=18, color="#F1C40F",
            markeredgecolor="black", markeredgewidth=1,
            zorder=4, label="GIGA post-refine")

    # annotate final values
    ax.annotate(f"GA final: {ga_mean[-1]:.4f}",
                xy=(ga_gens[-1], ga_mean[-1]),
                xytext=(ga_gens[-1] - 1.5, ga_mean[-1] + 0.02),
                fontsize=9, fontweight="bold", color="#E74C3C",
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))
    ax.annotate(f"GIGA final: {giga_mean[-1]:.4f}",
                xy=(giga_gens[-1], giga_mean[-1]),
                xytext=(giga_gens[-1] - 2, giga_mean[-1] - 0.03),
                fontsize=9, fontweight="bold", color="#2E86C1",
                arrowprops=dict(arrowstyle="->", color="#2E86C1", lw=1.2))

    ax.set_title("Average Convergence: GIGA vs Standard GA\n"
                 "(Shaded region = ±1 std dev across datasets)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness (lower is better)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    path = os.path.join(out_dir, "Convergence_average.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
# ── Output directory ──
RESULTS_DIR = "Final_Results_GIGA"
os.makedirs(os.path.join(RESULTS_DIR, "graphs"), exist_ok=True)

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
    "Decision Tree":  lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest":  lambda: RandomForestClassifier(n_estimators=30, random_state=42),
    "SVM":            lambda: SVC(probability=True, random_state=42),
    "XGBoost":        lambda: XGBClassifier(eval_metric='logloss', random_state=42),
}

FS_METHODS = ["Before", "GWO", "WHO", "FHO", "GA", "GIGA"]

results        = []   # [dataset, model, fs_method, accuracy, f1, n_features]
convergence_db = {}   # {dataset: {"GA": [...], "GIGA": [...]}}

for dname, path in datasets.items():
    if not os.path.exists(path):
        print(f"Skipping {dname} — file not found")
        continue
    try:
        X_raw, y = load_and_preprocess(path)
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.3, random_state=42)

        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        dim  = X_tr.shape[1]
        print(f"\n{dname}  (dim={dim})")

        # --- GA and GIGA now return (mask, convergence_history) ---
        ga_mask,   ga_conv   = GA(X_tr, X_te, y_train, y_test, dim)
        giga_mask, giga_conv = GIGA(X_tr, X_te, y_train, y_test, dim)

        convergence_db[dname] = {"GA": ga_conv, "GIGA": giga_conv}

        masks = {
            "Before": np.ones(dim, dtype=bool),
            "GWO":    GWO(X_tr, X_te, y_train, y_test, dim),
            "WHO":    WHO(X_tr, X_te, y_train, y_test, dim),
            "FHO":    FHO(X_tr, X_te, y_train, y_test, dim),
            "GA":     ga_mask,
            "GIGA":   giga_mask,
        }

        for mname, get_model in models_dict.items():
            for fsname, mask in masks.items():
                if mask.sum() == 0:
                    continue
                clf   = get_model()
                clf.fit(X_tr[:, mask], y_train)
                preds = clf.predict(X_te[:, mask])
                acc   = accuracy_score(y_test, preds)
                f1    = f1_score(y_test, preds, average='weighted')
                results.append([dname, mname, fsname, acc, f1, int(mask.sum())])
                print(f"  [{mname:15}] {fsname:7} Acc:{acc:.4f}  F1:{f1:.4f}  Feat:{mask.sum()}")

    except Exception as e:
        print(f"Error with {dname}: {e}")


# =============================================================================
# BUILD DATAFRAME
# =============================================================================
df_res   = pd.DataFrame(results,
                        columns=["Dataset", "Model", "FS",
                                 "Accuracy", "F1", "NFeatures"])
DATASETS = list(df_res["Dataset"].unique())
MODELS   = ["Decision Tree", "XGBoost", "SVM", "Random Forest"]


# =============================================================================
# MATPLOTLIB GRAPHS
# =============================================================================
FS_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000", "#7030A0", "#FFD700"]

def plot_grouped_bar(metric_col, metric_label, filename):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    n_ds  = len(DATASETS)
    n_fs  = len(FS_METHODS)
    x     = np.arange(n_ds)
    width = 0.13

    for ax_idx, model in enumerate(MODELS):
        ax  = axes[ax_idx]
        sub = df_res[df_res["Model"] == model]

        for fi, fs in enumerate(FS_METHODS):
            vals = []
            for ds in DATASETS:
                row = sub[(sub["Dataset"] == ds) & (sub["FS"] == fs)]
                vals.append(float(row[metric_col].values[0])
                            if not row.empty else 0.0)

            offset   = (fi - n_fs / 2 + 0.5) * width
            is_giga  = (fs == "GIGA")
            ax.bar(x + offset, vals, width,
                   color=FS_COLORS[fi],
                   edgecolor="black" if is_giga else "gray",
                   linewidth=1.5 if is_giga else 0.5,
                   label=fs,
                   zorder=3)

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{metric_label} — All Classifiers vs Feature Selection Methods",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_average_line(metric_col, metric_label, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax_idx, model in enumerate(MODELS):
        ax  = axes[ax_idx]
        sub = df_res[df_res["Model"] == model]

        avgs = []
        for fs in FS_METHODS:
            vals = sub[sub["FS"] == fs][metric_col].values
            avgs.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)

        ax.plot(FS_METHODS, avgs, color="#2E75B6",
                linewidth=2, marker="o", markersize=7,
                markerfacecolor="white", markeredgewidth=2,
                zorder=3)

        giga_idx = FS_METHODS.index("GIGA")
        ax.plot(giga_idx, avgs[giga_idx],
                marker="*", markersize=16,
                color="#FFD700", markeredgecolor="black",
                markeredgewidth=0.8, zorder=4, label="GIGA")

        for i, (fs, v) in enumerate(zip(FS_METHODS, avgs)):
            ax.text(i, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if fs == "GIGA" else "normal")

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(FS_METHODS)))
        ax.set_xticklabels(FS_METHODS, fontsize=10)
        ax.set_ylabel(f"Avg {metric_label}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"Average {metric_label} per Classifier Across All Datasets",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


graph_dir = os.path.join(RESULTS_DIR, "graphs")
os.makedirs(graph_dir, exist_ok=True)
print("\n>>> Generating graphs...")

plot_grouped_bar("Accuracy", "Accuracy", os.path.join(graph_dir, "Accuracy_per_dataset.png"))
plot_average_line("Accuracy", "Accuracy", os.path.join(graph_dir, "Accuracy_average.png"))
plot_grouped_bar("F1",       "F1-Score", os.path.join(graph_dir, "F1Score_per_dataset.png"))
plot_average_line("F1",      "F1-Score", os.path.join(graph_dir, "F1Score_average.png"))

# --- NEW convergence plots ---
plot_convergence_per_dataset(convergence_db, out_dir=graph_dir)
plot_convergence_average(convergence_db, out_dir=graph_dir)


# =============================================================================
# EXCEL REPORT
# =============================================================================
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
    return Border(top=s if top else NO, bottom=s if bot else NO,
                  left=s if lft else NO, right=s if rgt else NO)

def fl(c):
    return PatternFill("solid", start_color=c, end_color=c)

def sc(cell, bold=False, fc="000000", bg=None, sz=11,
       ha="center", fmt=None, bord=None):
    cell.font      = Font(name="Arial", bold=bold, color=fc, size=sz)
    cell.alignment = Alignment(horizontal=ha, vertical="center")
    if bg:   cell.fill  = fl(bg)
    if fmt:  cell.number_format = fmt
    if bord: cell.border = bord


def write_metric_sheet(ws, metric_label, metric_col):
    NCOLS = len(FS_METHODS)
    TOTAL = NCOLS + 2

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=TOTAL)
    sc(ws.cell(row=1, column=1,
               value=f"{metric_label} Results — All Classifiers vs Feature Selection Methods"),
       bold=True, fc="FFFFFF", bg=C_TITLE, sz=13,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    ws.row_dimensions[1].height = 24

    cur_row  = 2
    avg_rows = []

    for model in MODELS:
        sub = df_res[df_res["Model"] == model]

        sc(ws.cell(row=cur_row, column=1, value="Dataset"),
           bold=True, fc="FFFFFF", bg=C_HDR, sz=11,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        sc(ws.cell(row=cur_row, column=2, value="Model"),
           bold=True, fc="FFFFFF", bg=C_HDR, sz=11,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        for ci, fs in enumerate(FS_METHODS, start=3):
            sc(ws.cell(row=cur_row, column=ci, value=fs),
               bold=True, fc="FFFFFF",
               bg=C_GIGA if fs == "GIGA" else C_HDR, sz=11,
               bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        cur_row += 1

        for ci in range(1, TOTAL + 1):
            is_giga = (ci >= 3 and FS_METHODS[ci - 3] == "GIGA")
            val = metric_label if ci >= 3 else ""
            sc(ws.cell(row=cur_row, column=ci, value=val),
               bold=True, fc="FFFFFF",
               bg=C_GIGA if is_giga else C_HDR, sz=10,
               bord=bdr(bot=True, lft=True, rgt=True))
        cur_row += 1

        for r_idx, dset in enumerate(DATASETS):
            row_bg = C_ALT1 if r_idx % 2 == 0 else C_ALT2
            sc(ws.cell(row=cur_row, column=1, value=dset),
               fc="1F4E78", bg=row_bg, ha="left", sz=11,
               bord=bdr(lft=True, rgt=True))
            sc(ws.cell(row=cur_row, column=2, value=model),
               bg=row_bg, ha="left", sz=11,
               bord=bdr(lft=True, rgt=True))
            for ci, fs in enumerate(FS_METHODS, start=3):
                row = sub[(sub["Dataset"] == dset) & (sub["FS"] == fs)]
                val = round(float(row[metric_col].values[0]), 4) \
                      if not row.empty else ""
                sc(ws.cell(row=cur_row, column=ci, value=val),
                   bg=C_GIGA if fs == "GIGA" else row_bg,
                   sz=11, fmt="0.0000",
                   bord=bdr(lft=True, rgt=True))
            cur_row += 1

        sc(ws.cell(row=cur_row, column=1, value=""),
           bg=C_AVG,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
        sc(ws.cell(row=cur_row, column=2, value="AVERAGE"),
           bold=True, bg=C_AVG, ha="left", sz=11,
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

        data_start = cur_row - len(DATASETS)
        data_end   = cur_row - 1
        for ci in range(3, TOTAL + 1):
            vals = [ws.cell(row=r, column=ci).value
                    for r in range(data_start, data_end + 1)
                    if isinstance(ws.cell(row=r, column=ci).value, (int, float))]
            avg_val = round(sum(vals) / len(vals), 6) if vals else ""
            sc(ws.cell(row=cur_row, column=ci, value=avg_val),
               bold=True,
               bg=C_GIGA if FS_METHODS[ci - 3] == "GIGA" else C_AVG,
               sz=11, fmt="0.000000",
               bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

        avg_rows.append(cur_row)
        cur_row += 3

    sc(ws.cell(row=cur_row, column=1, value="Grand"),
       bold=True, fc="FFFFFF", bg=C_GRAND, sz=11,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    sc(ws.cell(row=cur_row, column=2, value="AVERAGE"),
       bold=True, fc="FFFFFF", bg=C_GRAND, ha="left", sz=11,
       bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))
    for ci in range(3, TOTAL + 1):
        vals = [ws.cell(row=ar, column=ci).value
                for ar in avg_rows
                if isinstance(ws.cell(row=ar, column=ci).value, (int, float))]
        grand = round(sum(vals) / len(vals), 6) if vals else ""
        sc(ws.cell(row=cur_row, column=ci, value=grand),
           bold=True, fc="FFFFFF",
           bg=C_GIGA if FS_METHODS[ci - 3] == "GIGA" else C_GRAND,
           sz=11, fmt="0.000000",
           bord=bdr(top=True, bot=True, lft=True, rgt=True, med=True))

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 17
    for ci in range(3, TOTAL + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 12
    ws.freeze_panes = "C3"


wb = Workbook()
wb.remove(wb.active)

for metric_label, metric_col in [("Accuracy", "Accuracy"),
                                  ("F1-Score", "F1")]:
    ws = wb.create_sheet(title=metric_label)
    write_metric_sheet(ws, metric_label, metric_col)

out_path = os.path.join(RESULTS_DIR, "Defect_Prediction_Results.xlsx")
wb.save(out_path)
print(f"\nDone! -> {out_path}")
print(f"Sheets : {wb.sheetnames}")
print(f"Results: {len(results)} rows")
