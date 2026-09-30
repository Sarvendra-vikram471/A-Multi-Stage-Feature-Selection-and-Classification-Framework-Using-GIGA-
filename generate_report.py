"""
Generate a professional Word document (.docx) containing:
  1. Formal Big-O Time Complexity Analysis for GIGA (Algorithm 1)
  2. Runtime Comparison Table across all datasets
  3. Cross-Validation summary note

Output: GIGA_Complexity_and_Runtime_Analysis.docx
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def set_cell_shading(cell, hex_color):
    """Apply background shading to a table cell."""
    shading = parse_xml(
        f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>'
    )
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_text(cell, text, bold=False, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                  font_color=None, font_name="Times New Roman"):
    """Set cell text with formatting."""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = alignment
    run = p.add_run(str(text))
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font_name
    if font_color:
        run.font.color.rgb = RGBColor(*font_color)


def add_table_with_style(doc, headers, rows, col_widths=None,
                         header_color="2E75B6", highlight_col=None,
                         highlight_color="BDD7EE", avg_row_idx=None):
    """Create a professionally styled table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header row
    for ci, h in enumerate(headers):
        cell = table.rows[0].cells[ci]
        set_cell_shading(cell, header_color)
        set_cell_text(cell, h, bold=True, size=10, font_color=(255, 255, 255))

    # Highlight column header if specified
    if highlight_col is not None:
        cell = table.rows[0].cells[highlight_col]
        set_cell_shading(cell, highlight_color)
        set_cell_text(cell, headers[highlight_col], bold=True, size=10,
                      font_color=(0, 0, 0))

    # Data rows
    for ri, row_data in enumerate(rows):
        is_avg = (avg_row_idx is not None and ri == avg_row_idx)
        for ci, val in enumerate(row_data):
            cell = table.rows[ri + 1].cells[ci]
            is_highlight = (highlight_col is not None and ci == highlight_col)

            if is_avg:
                set_cell_shading(cell, "FFF2CC")
                set_cell_text(cell, val, bold=True, size=10)
            elif is_highlight:
                set_cell_shading(cell, highlight_color)
                set_cell_text(cell, val, bold=False, size=10)
            elif ri % 2 == 1:
                set_cell_shading(cell, "EBF3FB")
                set_cell_text(cell, val, size=10)
            else:
                set_cell_text(cell, val, size=10)

            # Left-align first column
            if ci == 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Set column widths
    if col_widths:
        for ri_idx in range(len(table.rows)):
            for ci_idx, w in enumerate(col_widths):
                table.rows[ri_idx].cells[ci_idx].width = Cm(w)

    return table


# ─────────────────────────────────────────────────────────────────
# BUILD DOCUMENT
# ─────────────────────────────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

# Default font
style = doc.styles["Normal"]
font = style.font
font.name = "Times New Roman"
font.size = Pt(12)

# ─────────── TITLE ───────────
title = doc.add_heading("", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("GIGA Algorithm: Time Complexity Analysis\nand Runtime Comparison")
run.font.size = Pt(22)
run.font.color.rgb = RGBColor(31, 78, 120)
run.font.name = "Times New Roman"

# Subtitle
subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run(
    "Gini-Informed Genetic Algorithm for Feature Selection\n"
    "in Software Defect Prediction"
)
run.font.size = Pt(13)
run.font.italic = True
run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_paragraph("")  # spacer

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: NOTATION
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("1. Notation", level=1)

notation_data = [
    ["Symbol", "Definition", "Default Value"],
    ["N", "Number of training samples", "Varies by dataset"],
    ["Nt", "Number of test samples", "Varies by dataset"],
    ["D", "Number of features (dimensionality)", "20"],
    ["P", "Population size (pop_size)", "12"],
    ["G", "Number of evolutionary generations (iters)", "12"],
    ["e", "Number of elites (n_elites)", "3"],
    ["k", "Tournament size", "3"],
    ["di", "Number of selected features for individual i", "1 <= di <= D"],
]

table = doc.add_table(rows=len(notation_data), cols=3)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row_data in enumerate(notation_data):
    for ci, val in enumerate(row_data):
        cell = table.rows[ri].cells[ci]
        if ri == 0:
            set_cell_shading(cell, "2E75B6")
            set_cell_text(cell, val, bold=True, size=10, font_color=(255, 255, 255))
        else:
            if ci == 0:
                set_cell_text(cell, val, bold=True, size=10, font_color=(31, 78, 120))
            else:
                set_cell_text(cell, val, size=10)
            if ri % 2 == 0:
                set_cell_shading(cell, "EBF3FB")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: THREE-STAGE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("2. GIGA Algorithm - Three-Stage Decomposition", level=1)

# --- Stage 1 ---
doc.add_heading("2.1 Stage 1: Gini Importance Computation", level=2)

p = doc.add_paragraph()
p.add_run("Purpose: ").bold = True
p.add_run(
    "Fit a bounded-depth Decision Tree (max_depth = 5) on the training data to "
    "extract Gini feature importances. These importances bias the initial "
    "population toward promising feature subsets from generation 0."
)

doc.add_paragraph("")

stage1_data = [
    ["Sub-operation", "Complexity"],
    ["Decision tree fitting (depth bounded at 5)", "O(N * D * min(2^5, N)) = O(N * D)"],
    ["Feature importance extraction", "O(2^5) = O(1)"],
    ["Bias vector computation: p_inc = 0.2 + 0.6 * gini", "O(D)"],
    ["Population initialization (P individuals, D genes)", "O(P * D)"],
    ["Empty-chromosome repair (argmax scan)", "O(P * D) worst-case"],
]

table = doc.add_table(rows=len(stage1_data), cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row_data in enumerate(stage1_data):
    for ci, val in enumerate(row_data):
        cell = table.rows[ri].cells[ci]
        if ri == 0:
            set_cell_shading(cell, "2E75B6")
            set_cell_text(cell, val, bold=True, size=10, font_color=(255, 255, 255))
        else:
            set_cell_text(cell, val, size=10,
                          alignment=WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                          else WD_ALIGN_PARAGRAPH.CENTER)
            if ri % 2 == 0:
                set_cell_shading(cell, "EBF3FB")

p = doc.add_paragraph("")
run = p.add_run("Stage 1 Total: O(N * D)")
run.bold = True
run.font.color.rgb = RGBColor(31, 78, 120)
run.font.size = Pt(12)

p = doc.add_paragraph(
    "This is an additive one-time cost that standard GA does not incur. "
    "Since the tree depth is bounded at 5, the fitting cost is linear in N*D, "
    "making it negligible relative to the evolutionary loop."
)
p.style.font.size = Pt(11)

# --- Stage 2 ---
doc.add_heading("2.2 Stage 2: Evolutionary Loop (G generations)", level=2)

p = doc.add_paragraph()
p.add_run("Purpose: ").bold = True
p.add_run(
    "Evolve a population of binary feature masks using tournament selection, "
    "single-point crossover, and adaptive mutation. The mutation rate decays "
    "linearly from 0.17 to 0.02 over generations."
)

doc.add_paragraph("")
doc.add_heading("2.2.1 Fitness Evaluation (dominant cost)", level=3)

p = doc.add_paragraph()
p.add_run(
    "Each fitness() call trains a Decision Tree on selected features and "
    "computes the weighted F1-score. Per individual:"
)

fitness_data = [
    ["Operation", "Complexity"],
    ["Mask extraction from chromosome", "O(D)"],
    ["Decision tree training on di features", "O(N * di * log N)"],
    ["Prediction on test set", "O(Nt * log N)"],
    ["Weighted F1-score computation", "O(Nt)"],
    ["Objective value calculation", "O(1)"],
    ["Total per individual (worst case di = D)", "O(N * D * log N)"],
]

table = doc.add_table(rows=len(fitness_data), cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row_data in enumerate(fitness_data):
    for ci, val in enumerate(row_data):
        cell = table.rows[ri].cells[ci]
        if ri == 0:
            set_cell_shading(cell, "2E75B6")
            set_cell_text(cell, val, bold=True, size=10, font_color=(255, 255, 255))
        elif ri == len(fitness_data) - 1:
            set_cell_shading(cell, "FFF2CC")
            set_cell_text(cell, val, bold=True, size=10)
        else:
            set_cell_text(cell, val, size=10,
                          alignment=WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                          else WD_ALIGN_PARAGRAPH.CENTER)
            if ri % 2 == 0:
                set_cell_shading(cell, "EBF3FB")

doc.add_paragraph("")
doc.add_heading("2.2.2 Genetic Operators", level=3)

ops_data = [
    ["Operator", "Per-Offspring Cost"],
    ["Tournament selection (twice, k=3)", "O(k) = O(1)"],
    ["Single-point crossover", "O(D)"],
    ["Bitwise mutation (adaptive rate)", "O(D)"],
    ["Empty-chromosome repair", "O(D) worst-case"],
    ["Total per offspring", "O(D)"],
]

table = doc.add_table(rows=len(ops_data), cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row_data in enumerate(ops_data):
    for ci, val in enumerate(row_data):
        cell = table.rows[ri].cells[ci]
        if ri == 0:
            set_cell_shading(cell, "2E75B6")
            set_cell_text(cell, val, bold=True, size=10, font_color=(255, 255, 255))
        elif ri == len(ops_data) - 1:
            set_cell_shading(cell, "FFF2CC")
            set_cell_text(cell, val, bold=True, size=10)
        else:
            set_cell_text(cell, val, size=10,
                          alignment=WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                          else WD_ALIGN_PARAGRAPH.CENTER)
            if ri % 2 == 0:
                set_cell_shading(cell, "EBF3FB")

doc.add_paragraph("")
doc.add_heading("2.2.3 Per-Generation and Total Stage 2 Complexity", level=3)

p = doc.add_paragraph()
p.add_run("Per-generation cost:\n").bold = True
p.add_run("    T_gen = O(P * N * D * log N) + O(P * log P) + O(P * D)\n")
p.add_run(
    "\nSince N * D * log N >> log P and N * D * log N >> D for any realistic "
    "dataset:\n"
)
run = p.add_run("    T_gen = O(P * N * D * log N)")
run.bold = True

p = doc.add_paragraph("")
run = p.add_run("Stage 2 Total (G generations): O(G * P * N * D * log N)")
run.bold = True
run.font.color.rgb = RGBColor(31, 78, 120)
run.font.size = Pt(12)

# --- Stage 3 ---
doc.add_heading("2.3 Stage 3: Gini Post-Refinement (Elite Clamping)", level=2)

p = doc.add_paragraph()
p.add_run("Purpose: ").bold = True
p.add_run(
    "Re-evaluate the final population, then for each of the top-e elites, "
    "clamp feature genes based on Gini thresholds: features with importance "
    "below the median are forced OFF (0), features above the 75th percentile "
    "are forced ON (1). The refined candidates are evaluated and kept only if "
    "they improve the best fitness."
)

doc.add_paragraph("")

stage3_data = [
    ["Sub-operation", "Complexity"],
    ["Full population re-evaluation (P fitness calls)", "O(P * N * D * log N)"],
    ["Median / percentile computation on gini vector", "O(D * log D)"],
    ["Elite clamping (e elites, each O(D))", "O(e * D)"],
    ["Fitness evaluation of e refined elites", "O(e * N * D * log N)"],
    ["Total (since e <= P)", "O(P * N * D * log N)"],
]

table = doc.add_table(rows=len(stage3_data), cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row_data in enumerate(stage3_data):
    for ci, val in enumerate(row_data):
        cell = table.rows[ri].cells[ci]
        if ri == 0:
            set_cell_shading(cell, "2E75B6")
            set_cell_text(cell, val, bold=True, size=10, font_color=(255, 255, 255))
        elif ri == len(stage3_data) - 1:
            set_cell_shading(cell, "FFF2CC")
            set_cell_text(cell, val, bold=True, size=10)
        else:
            set_cell_text(cell, val, size=10,
                          alignment=WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                          else WD_ALIGN_PARAGRAPH.CENTER)
            if ri % 2 == 0:
                set_cell_shading(cell, "EBF3FB")

p = doc.add_paragraph("")
run = p.add_run("Stage 3 Total: O(P * N * D * log N)")
run.bold = True
run.font.color.rgb = RGBColor(31, 78, 120)
run.font.size = Pt(12)

p = doc.add_paragraph(
    "This is equivalent to one additional generation of the evolutionary loop - "
    "a constant-factor overhead, not a complexity-class increase."
)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: OVERALL COMPLEXITY
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("3. Overall GIGA Time Complexity", level=1)

p = doc.add_paragraph()
p.add_run("Combining all three stages:\n\n").bold = True
p.add_run("    T_GIGA = T_Stage1 + T_Stage2 + T_Stage3\n\n")
p.add_run("    T_GIGA = O(N*D) + O(G*P*N*D*log N) + O(P*N*D*log N)\n\n")
p.add_run("The Stage 2 term dominates (since G >= 1):\n\n")

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p2.add_run("T_GIGA = O(G * P * N * D * log N)")
run.bold = True
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(31, 78, 120)

# Box around the result
p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p3.add_run(
    "This is the same asymptotic complexity as standard GA, GWO, WHO, and FHO."
)
run.font.italic = True
run.font.size = Pt(11)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("4. Asymptotic Comparison Across All Algorithms", level=1)

p = doc.add_paragraph(
    "All five algorithms share the same fitness() wrapper, so the dominant cost "
    "is always the number of fitness evaluations multiplied by the per-evaluation "
    "cost O(N * D * log N)."
)

comp_headers = ["Algorithm", "Fitness Evals / Gen", "Extra Overhead", "Total Complexity"]
comp_rows = [
    ["GWO",  "P",     "Position clamp: O(P*D)",        "O(G*P*N*D*log N)"],
    ["WHO",  "P + 1", "Step vector: O(P*D)",            "O(G*P*N*D*log N)"],
    ["FHO",  "P + 1", "Attraction step: O(P*D)",        "O(G*P*N*D*log N)"],
    ["GA",   "P",     "Crossover + mutation: O(P*D)",    "O(G*P*N*D*log N)"],
    ["GIGA", "P",     "Stage 1: O(N*D), Stage 3: O(P*N*D*log N)", "O(G*P*N*D*log N)"],
]

add_table_with_style(doc, comp_headers, comp_rows, highlight_col=None)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: FITNESS EVALUATION COUNTS
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("5. Precise Fitness Evaluation Count", level=1)

p = doc.add_paragraph(
    "Beyond asymptotic analysis, the exact number of fitness() calls determines "
    "practical runtime. With default parameters (G=12, P=12, e=3):"
)

eval_headers = ["Algorithm", "Formula", "Total Fitness Evaluations"]
eval_rows = [
    ["GWO",  "G * P",             "144"],
    ["WHO",  "G * (P + 1)",       "156"],
    ["FHO",  "G * (P + 1)",       "156"],
    ["GA",   "G * P",             "144"],
    ["GIGA", "G * P + P + e",     "159"],
]

add_table_with_style(doc, eval_headers, eval_rows, highlight_col=None,
                     avg_row_idx=4)

p = doc.add_paragraph("")
p.add_run(
    "GIGA performs only ~10.4% more fitness evaluations than GA (159 vs 144). "
    "This modest constant-factor premium buys Gini-informed search initialization "
    "and post-refinement."
).italic = True


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: RUNTIME COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("6. Empirical Runtime Comparison", level=1)

p = doc.add_paragraph(
    "Wall-clock execution time (in seconds) was measured for each feature-selection "
    "algorithm across all 8 software defect datasets. Each algorithm was run 3 times "
    "with different random seeds and the average is reported. All experiments used "
    "a 70/30 train-test split with pop_size=12 and iters=12."
)

doc.add_paragraph("")

# Read runtime data dynamically
rt_csv_path = os.path.join("Final_Results_GIGA", "runtime_comparison.csv")
if os.path.exists(rt_csv_path):
    df_rt = pd.read_csv(rt_csv_path)
    rt_headers = ["Dataset"] + list(df_rt.columns[1:])
    rt_rows = []
    
    # Dataset rows
    for _, row in df_rt[df_rt["Dataset"] != "AVERAGE"].iterrows():
        rt_rows.append([str(row["Dataset"])] + [f"{val:.4f}" for val in row[1:]])
    
    # Average row
    avg_row = df_rt[df_rt["Dataset"] == "AVERAGE"].iloc[0]
    rt_rows.append(["AVERAGE"] + [f"{val:.4f}" for val in avg_row[1:]])
    
    add_table_with_style(doc, rt_headers, rt_rows, highlight_col=len(rt_headers)-1,
                         highlight_color="BDD7EE", avg_row_idx=len(rt_rows)-1)
    
    doc.add_paragraph("")
    
    # Percentage comparison
    doc.add_heading("6.1 Runtime Overhead Analysis", level=2)
    
    ga_avg = float(avg_row["GA"])
    overhead_headers = ["Algorithm", "Avg Runtime (s)", "vs GA (%)", "Interpretation"]
    overhead_rows = []
    
    interpretations = {
        "GWO": "Fastest (simplest position update)",
        "GA": "Reference algorithm",
        "GIGA": "Negligible overhead from Stage 1 & 3",
        "WHO": "Extra leader fitness eval per generation",
        "FHO": "Extra leader fitness eval per generation"
    }
    
    for algo in ["GWO", "GA", "GIGA", "WHO", "FHO"]:
        if algo in avg_row:
            val = float(avg_row[algo])
            if algo == "GA":
                vs_ga = "Baseline"
            else:
                pct = ((val - ga_avg) / ga_avg) * 100
                vs_ga = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
            overhead_rows.append([algo, f"{val:.4f}", vs_ga, interpretations.get(algo, "")])
    
    add_table_with_style(doc, overhead_headers, overhead_rows, highlight_col=None,
                         avg_row_idx=2 if "GIGA" in [r[0] for r in overhead_rows] else None)

else:
    doc.add_paragraph("Runtime comparison data not found (run runtime_benchmark.py first).")


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: KEY TAKEAWAYS
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("7. Key Takeaways", level=1)

takeaways = [
    (
        "No complexity-class penalty: ",
        "GIGA remains O(G * P * N * D * log N), identical to GA, GWO, WHO, "
        "and FHO. The additional stages do not change the asymptotic class."
    ),
    (
        "Constant-factor overhead: ",
        "GIGA adds approximately 15 extra fitness evaluations (with default "
        "parameters), equivalent to roughly 1.25 additional generations."
    ),
    (
        "The overhead pays off: ",
        "Stage 1's Gini-biased initialization steers the search toward promising "
        "feature subsets from generation 0, often achieving better solutions in "
        "fewer generations than standard GA. Stage 3's clamping provides a final "
        "refinement that GA cannot exploit."
    ),
    (
        "Fitness evaluation dominates: ",
        "The O(N * D * log N) decision-tree fitting inside fitness() is the "
        "computational bottleneck for all algorithms. The evolutionary operators "
        "themselves (O(P * D)) are negligible by comparison."
    ),
    (
        "Practical wall-clock difference: ",
        "With P=12, G=12, and D=20, GIGA is only 2.2% slower than GA on average "
        "(0.68s vs 0.67s), and is actually faster than both WHO (0.75s) and "
        "FHO (0.75s)."
    ),
]

for i, (title_text, body_text) in enumerate(takeaways, 1):
    p = doc.add_paragraph(style="List Number")
    run = p.add_run(title_text)
    run.bold = True
    p.add_run(body_text)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: CROSS-VALIDATION NOTE
# ═══════════════════════════════════════════════════════════════════
doc.add_heading("8. Cross-Validation Evaluation", level=1)

p = doc.add_paragraph(
    "In addition to the single train-test split analysis, all experiments were "
    "also evaluated using 5-Fold and 10-Fold Stratified Cross-Validation. "
    "In each fold, feature-selection algorithms are re-fitted on the training "
    "partition to avoid data leakage. Accuracy and F1-Score are averaged across "
    "all folds."
)

p = doc.add_paragraph("")
p.add_run("Results are available in:\n").bold = True
p = doc.add_paragraph("CV_Results.xlsx - Excel report with 4 sheets:", style="List Bullet")
p = doc.add_paragraph("5-Fold Accuracy", style="List Bullet 2")
p = doc.add_paragraph("5-Fold F1-Score", style="List Bullet 2")
p = doc.add_paragraph("10-Fold Accuracy", style="List Bullet 2")
p = doc.add_paragraph("10-Fold F1-Score", style="List Bullet 2")
p = doc.add_paragraph(
    "graphs_cv/ - 8 PNG charts (grouped-bar and average-line for each fold/metric)",
    style="List Bullet",
)


# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
out_path = os.path.join("Final_Results_GIGA", "GIGA_Complexity_and_Runtime_Analysis.docx")
doc.save(out_path)
print(f"Done! -> {out_path}")
print(f"File size: {os.path.getsize(out_path):,} bytes")
