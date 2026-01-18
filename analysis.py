import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import sys
import os

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
FILE_PATH = 'data/results/master_results.csv'
REPORT_FILE = 'thesis_expert_analysis_v15.txt'
PLOT_FOLDER = 'Thesis_Plots_Expert_v15'

# Academic/Paper Styling
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.titleweight': 'bold',
    'font.family': 'sans-serif'
})

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

print(f"--- 🎓 THESIS MASTER ENGINE v15 (COMPLETE) ---")

# ==========================================
# 2. DATA PREP
# ==========================================
if not os.path.exists(FILE_PATH):
    print(f"❌ ERROR: File not found at {FILE_PATH}")
    sys.exit()

df = pd.read_csv(FILE_PATH, low_memory=False)

# Numeric Cleanup
metric_cols = ['ndcg@10', 'recall@10', 'precision@10', 'map@10', 'mrr@10',
               'chunking_time_s', 'indexing_duration_s', 'chunk_avg_chars',
               'num_chunks']

# Add Top-K columns
for k in [1, 3, 5, 10, 20]:
    metric_cols.extend([f'recall@{k}', f'ndcg@{k}', f'precision@{k}', f'map@{k}'])

for col in metric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# --- Feature 1: Strategy Classification ---
def get_strat(row):
    exp = str(row['experiment']).lower()
    if 'semantic' in exp: return 'Semantic'
    if 'recursive' in exp: return 'Recursive'
    if 'fixed' in exp: return 'Fixed'
    if 'sentence' in exp: return 'Sentence'
    return 'Other'


df['Strategy'] = df.apply(get_strat, axis=1)


# --- Feature 2: Parameter Extraction ---
def get_params(row):
    exp = str(row['experiment']).lower()
    strat = get_strat(row)
    size, overlap, thresh = np.nan, np.nan, np.nan

    if strat in ['Fixed', 'Recursive']:
        nums = re.findall(r'\d+', exp)
        if len(nums) >= 2:
            size = int(nums[0])
            overlap = int(nums[1])

    if strat == 'Semantic':
        match = re.search(r't(\d?\.?\d+)', exp)
        if match: thresh = float(match.group(1))

    return pd.Series([size, overlap, thresh])


df[['Param_Size', 'Param_Overlap', 'Param_Threshold']] = df.apply(get_params, axis=1)

# --- Feature 3: Size Bins ---
bins = [0, 200, 400, 600, 800, 1000, 2000, 10000]
labels = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'Huge']
df['Size_Bin'] = pd.cut(df['chunk_avg_chars'], bins=bins, labels=labels)

# --- Feature 4: Efficiency ---
df['Score_Efficiency'] = (df['ndcg@10'] / df['chunk_avg_chars']) * 100

# ==========================================
# 3. GENERATE EXPERT REPORT
# ==========================================
print(f"Writing Expert Analysis -> {REPORT_FILE}")

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=== THESIS EXPERT ANALYSIS v15 ===\n")
    f.write("Includes Embedding Model Impact, Parameters, and Expert Metrics.\n\n")

    # --- 1. FULL METRIC BREAKDOWN (H1) ---
    f.write("### 1. Comprehensive Metric Analysis (H1)\n")
    metrics = ['ndcg@10', 'recall@10', 'precision@10', 'map@10', 'mrr@10']
    metrics = [m for m in metrics if m in df.columns]

    summary = df.groupby('Strategy')[metrics].agg(['mean', 'std'])
    f.write(summary.to_markdown())

    # --- 2. EMBEDDING MODEL IMPACT (H1b) ---
    if 'embedding_model' in df.columns:
        f.write("\n\n### 2. Embedding Model Impact (Deep Dive)\n")
        f.write("Does the choice of embedding model change the winner?\n")

        # Calculate scores per model per strategy
        model_pivot = df.pivot_table(index='embedding_model', columns='Strategy', values='ndcg@10', aggfunc='mean')
        f.write(model_pivot.to_markdown())

        f.write(
            "\n\n-> Expert Note: Compare if 'Semantic' performs better with specific models (e.g. BGE vs MiniLM).\n")

    # --- 3. PARAMETER DEEP DIVE ---
    f.write("\n\n### 3. Parameter Sensitivity\n")

    # Threshold (Semantic)
    f.write("\nA. Semantic Threshold:\n")
    t_perf = df.groupby('Param_Threshold')[['ndcg@10', 'recall@10', 'precision@10']].mean()
    f.write(t_perf.to_markdown())

    # Overlap (Fixed)
    f.write("\n\nB. Overlap Impact:\n")
    o_perf = df.groupby('Param_Overlap')[['ndcg@10', 'recall@10']].mean()
    f.write(o_perf.to_markdown())

    # --- 4. DIFFICULTY IMPACT (H4) ---
    if 'config_difficulty' in df.columns:
        f.write("\n\n### 4. Difficulty Analysis (H4)\n")
        diff_pivot = df.pivot_table(index='Strategy', columns='config_difficulty', values='ndcg@10')
        f.write(diff_pivot.to_markdown())

# ==========================================
# 4. PLOTTING (12+ PLOTS)
# ==========================================
print("Generating Plots...")
palette = {"Fixed": "#3498db", "Semantic": "#e74c3c", "Recursive": "#2ecc71", "Sentence": "#9b59b6", "Other": "gray"}

# 1. H1 Boxplot (NDCG)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Strategy', y='ndcg@10', palette=palette, showfliers=False)
plt.title("H1: Retrieval Quality Distribution (NDCG@10)")
plt.savefig(f"{PLOT_FOLDER}/01_H1_Overview_NDCG.png")
plt.close()

# 2. Embedding Model Impact (Bar Chart) - REQUESTED
if 'embedding_model' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='embedding_model', y='ndcg@10', hue='Strategy', palette=palette, errorbar=None)
    plt.title("Impact of Embedding Model on Strategy Performance")
    plt.xticks(rotation=15, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/02_Embedding_Model_Impact.png")
    plt.close()

# 3. Embedding Model x Strategy (Line Chart for Interaction) - REQUESTED
if 'embedding_model' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.pointplot(data=df, x='embedding_model', y='ndcg@10', hue='Strategy', palette=palette, dodge=True,
                  markers=['o', 's', 'D', '^', 'x'], linestyles='-')
    plt.title("Interaction: Embedding Model vs Strategy Effectiveness")
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/03_Embedding_Interaction_Line.png")
    plt.close()

# 4. H2 Size Curve
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Size_Bin', y='ndcg@10', hue='Strategy', palette=palette, marker='o', errorbar=None)
plt.title("H2: Size Effect on Performance (NDCG@10)")
plt.savefig(f"{PLOT_FOLDER}/04_H2_Size_Curve.png")
plt.close()

# 5. Semantic Threshold Analysis
sem = df[df['Strategy'] == 'Semantic']
if not sem.empty:
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sem, x='Param_Threshold', y='ndcg@10', marker='o', color='red', label='NDCG')
    plt.title("Parameter Analysis: Semantic Threshold vs NDCG")
    plt.savefig(f"{PLOT_FOLDER}/05_Param_Threshold.png")
    plt.close()

# 6. Fixed Overlap Analysis
fix = df[df['Strategy'].isin(['Fixed', 'Recursive'])]
if not fix.empty:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=fix, x='Param_Overlap', y='ndcg@10', hue='Strategy', palette=palette, errorbar=None)
    plt.title("Parameter Analysis: Overlap Impact")
    plt.savefig(f"{PLOT_FOLDER}/06_Param_Overlap.png")
    plt.close()

# 7. Precision-Recall Curve (Top-K)
k_vals = [1, 3, 5, 10, 20]
pr_data = []
for strat in df['Strategy'].unique():
    strat_df = df[df['Strategy'] == strat]
    for k in k_vals:
        if f'recall@{k}' in strat_df and f'precision@{k}' in strat_df:
            r = strat_df[f'recall@{k}'].mean()
            p = strat_df[f'precision@{k}'].mean()
            pr_data.append({'Strategy': strat, 'k': k, 'Recall': r, 'Precision': p})
pr_df = pd.DataFrame(pr_data)

plt.figure(figsize=(10, 8))
sns.lineplot(data=pr_df, x='Recall', y='Precision', hue='Strategy', palette=palette, marker='o', linewidth=2.5,
             sort=False)
plt.title("Precision-Recall Trade-off (Top-K Analysis)")
plt.grid(True)
plt.savefig(f"{PLOT_FOLDER}/07_Precision_Recall_Curve.png")
plt.close()

# 8. H3 Efficiency Scatter
agg = df.groupby(['experiment', 'Strategy'])[['indexing_duration_s', 'ndcg@10']].mean().reset_index()
plt.figure(figsize=(9, 6))
sns.scatterplot(data=agg, x='indexing_duration_s', y='ndcg@10', hue='Strategy', palette=palette, s=150, alpha=0.8)
plt.xscale('log')
plt.title("H3: Cost vs Quality (Efficiency Frontier)")
plt.savefig(f"{PLOT_FOLDER}/08_H3_Efficiency.png")
plt.close()

# 9. Difficulty Impact (H4)
if 'config_difficulty' in df.columns:
    try:
        # Filter for known difficulties
        diff_df = df[df['config_difficulty'].isin(['Easy', 'Moderate', 'Hard'])]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=diff_df, x='config_difficulty', y='ndcg@10', hue='Strategy', palette=palette,
                    order=['Easy', 'Moderate', 'Hard'], errorbar=None)
        plt.title("H4: Performance Across Difficulty Levels")
        plt.savefig(f"{PLOT_FOLDER}/09_H4_Complexity.png")
        plt.close()
    except:
        pass

# 10. Top-K Saturation
k_cols = [c for c in df.columns if 'recall@' in c]
melted = df.melt(id_vars='Strategy', value_vars=k_cols, var_name='K', value_name='Recall')
melted['K'] = melted['K'].str.extract(r'(\d+)').astype(int)
plt.figure(figsize=(10, 6))
sns.lineplot(data=melted, x='K', y='Recall', hue='Strategy', palette=palette, marker='d', errorbar=None)
plt.title("Retrieval Depth: Recall @ K")
plt.xticks([1, 3, 5, 10, 20])
plt.savefig(f"{PLOT_FOLDER}/10_TopK_Recall.png")
plt.close()

# 11. MRR Analysis
if 'mrr@10' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Strategy', y='mrr@10', palette=palette, errorbar=None)
    plt.title("MRR@10: First Relevant Result Rank")
    plt.savefig(f"{PLOT_FOLDER}/11_MRR_Analysis.png")
    plt.close()

# 12. Token Economics
plt.figure(figsize=(9, 6))
sns.barplot(data=df, x='Strategy', y='Score_Efficiency', palette=palette, errorbar=None)
plt.title("Token Economics: NDCG per 100 Characters")
plt.ylabel("Efficiency Score")
plt.savefig(f"{PLOT_FOLDER}/12_Token_Economics.png")
plt.close()

print(f"✅ ALL DONE. Report and 12 Plots saved in '{PLOT_FOLDER}'.")