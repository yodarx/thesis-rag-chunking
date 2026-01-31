import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import sys
import os
import matplotlib.ticker as ticker

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
               'indexing_duration_s', 'chunk_avg_chars',
               'num_chunks', 'chunks_per_second', 'processing_time_seconds',
               'chunk_total_count']

# Add Top-K columns
for k in [1, 3, 5, 10, 20]:
    metric_cols.extend([f'recall@{k}', f'ndcg@{k}', f'precision@{k}', f'map@{k}'])

for col in metric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Time Handling (Seconds -> Hours) ---
# Ensure processing_time_seconds is populated (fallback to speed calculation if missing)
if ('processing_time_seconds' not in df.columns or df['processing_time_seconds'].isna().all()) and \
   'chunks_per_second' in df.columns and 'num_chunks' in df.columns:
    df['processing_time_seconds'] = df['num_chunks'] / df['chunks_per_second']
    df['processing_time_seconds'] = df['processing_time_seconds'].replace([np.inf, -np.inf], np.nan)

# Create Hour columns (Handle 0 to avoid Log errors later)
df['indexing_time_h'] = df['indexing_duration_s'] / 3600 if 'indexing_duration_s' in df.columns else np.nan
df['chunking_time_h'] = df['processing_time_seconds'] / 3600 if 'processing_time_seconds' in df.columns else np.nan
df['total_prep_time_h'] = df['indexing_time_h'].fillna(0) + df['chunking_time_h'].fillna(0)

# Replace 0 with small epsilon for log plots
df['indexing_time_h'] = df['indexing_time_h'].replace(0, np.nan)
df['chunking_time_h'] = df['chunking_time_h'].replace(0, np.nan)
df['total_prep_time_h'] = df['total_prep_time_h'].replace(0, np.nan)

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

    # --- 5. GERMAN SUMMARY ---
    f.write("\n\n### Vergleich der Retrieval-Leistung (NDCG, Recall, Precision, MAP, MRR @10) zwischen Fixed, Recursive, Sentence und Semantic Chunking.\n")
    comp_strat = ['Fixed', 'Recursive', 'Sentence', 'Semantic']
    comp_metrics = ['ndcg@10', 'recall@10', 'precision@10', 'map@10', 'mrr@10']
    cols_exist = [c for c in comp_metrics if c in df.columns]
    if cols_exist:
        german_summary = df[df['Strategy'].isin(comp_strat)].groupby('Strategy')[cols_exist].mean()
        f.write(german_summary.to_markdown())

# ==========================================
# 4. PLOTTING (12+ PLOTS)
# ==========================================
print("Generating Plots...")
palette = {"Fixed": "#3498db", "Semantic": "#e74c3c", "Recursive": "#2ecc71", "Sentence": "#9b59b6", "Other": "gray"}

# 1. H1 Boxplot (NDCG)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Strategy', y='ndcg@10', palette=palette, showfliers=False)
plt.title("H1: Verteilung der Retrieval-Qualität (NDCG@10)")
plt.xlabel("Strategie")
plt.ylabel("NDCG@10")
plt.savefig(f"{PLOT_FOLDER}/01_H1_Overview_NDCG.png")
plt.close()

# 2. Embedding Model Impact (Bar Chart) - REQUESTED
if 'embedding_model' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='embedding_model', y='ndcg@10', hue='Strategy', palette=palette, errorbar=None)
    plt.title("Einfluss des Embedding-Modells auf die Strategie-Leistung")
    plt.ylabel("NDCG@10")
    plt.xlabel("Embedding Modell")
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
    plt.title("Interaktion: Embedding-Modell vs. Strategie-Effektivität")
    plt.ylabel("NDCG@10")
    plt.xlabel("Embedding Modell")
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/03_Embedding_Interaction_Line.png")
    plt.close()

# 4. H2 Size Curve (Focus: Impact of Window Size)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Size_Bin', y='ndcg@10', hue='Strategy', palette=palette, marker='o', errorbar=None)
plt.title("H2: Einfluss der Fenstergröße auf NDCG@10 über Strategien hinweg")
plt.ylabel("NDCG@10")
plt.xlabel("Chunk-Größe (Bins)")
plt.savefig(f"{PLOT_FOLDER}/04_H2_Size_Curve.png")
plt.close()

# 5. Semantic Threshold Analysis
sem = df[df['Strategy'] == 'Semantic']
if not sem.empty:
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sem, x='Param_Threshold', y='ndcg@10', marker='o', color='red', label='NDCG')
    plt.title("Parameter-Analyse: Semantischer Schwellenwert vs. NDCG@10")
    plt.ylabel("NDCG@10")
    plt.xlabel("Schwellenwert (Threshold)")
    plt.savefig(f"{PLOT_FOLDER}/05_Param_Threshold.png")
    plt.close()

# 6. Fixed Overlap Analysis
fix = df[df['Strategy'].isin(['Fixed', 'Recursive'])]
if not fix.empty:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=fix, x='Param_Overlap', y='ndcg@10', hue='Strategy', palette=palette, errorbar=None)
    plt.title("Parameter-Analyse: Einfluss der Überlappung (Overlap)")
    plt.ylabel("NDCG@10")
    plt.xlabel("Overlap (Zeichen)")
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
plt.title("PRECISION-RECALL Verhältnis (Top-K Analyse)")
plt.ylabel("PRECISION")
plt.xlabel("RECALL")
plt.grid(True)
plt.savefig(f"{PLOT_FOLDER}/07_Precision_Recall_Curve.png")
plt.close()

# 8. H3 Efficiency Scatter
if 'indexing_time_h' in df.columns:
    agg = df.groupby(['experiment', 'Strategy'])[['indexing_time_h', 'ndcg@10']].mean().reset_index()
    agg = agg[agg['indexing_time_h'] > 0]
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=agg, x='indexing_time_h', y='ndcg@10', hue='Strategy', palette=palette, s=150, alpha=0.8)

    plt.title("H3: Kosten vs. Qualität (Effizienz-Grenze)")
    plt.ylabel("NDCG@10")
    plt.xlabel("Indexierungsdauer (Stunden)")
    plt.grid(True, alpha=0.5)
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
        plt.title("H4: Leistung über Schwierigkeitsgrade hinweg")
        plt.ylabel("NDCG@10")
        plt.xlabel("Schwierigkeit")
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
plt.title("Retrieval-Tiefe: RECALL @ K")
plt.ylabel("RECALL")
plt.xlabel("K (Anzahl Dokumente)")
plt.xticks([1, 3, 5, 10, 20])
plt.savefig(f"{PLOT_FOLDER}/10_TopK_Recall.png")
plt.close()

# 11. MRR Analysis
if 'mrr@10' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Strategy', y='mrr@10', palette=palette, errorbar=None)
    plt.title("MRR@10: Rang des ersten relevanten Ergebnisses")
    plt.ylabel("MRR@10")
    plt.xlabel("Strategie")
    plt.savefig(f"{PLOT_FOLDER}/11_MRR_Analysis.png")
    plt.close()

# 12. Token Economics
plt.figure(figsize=(9, 6))
sns.barplot(data=df, x='Strategy', y='Score_Efficiency', palette=palette, errorbar=None)
plt.title("Token-Ökonomie: NDCG pro 100 Zeichen")
plt.ylabel("Effizienz-Score (NDCG/Größe)")
plt.xlabel("Strategie")
plt.savefig(f"{PLOT_FOLDER}/12_Token_Economics.png")
plt.close()

# 13. German Comparison Plot
comp_strategies = ['Fixed', 'Recursive', 'Sentence', 'Semantic']
comp_df = df[df['Strategy'].isin(comp_strategies)]
plot_metrics = ['ndcg@10', 'recall@10', 'precision@10', 'map@10', 'mrr@10']
valid_metrics = [m for m in plot_metrics if m in comp_df.columns]

if not comp_df.empty and valid_metrics:
    plt.figure(figsize=(12, 7))
    melted_comp = comp_df.melt(id_vars='Strategy', value_vars=valid_metrics,
                               var_name='Metric', value_name='Score')
    # Cleanup metric names for display
    melted_comp['Metric'] = melted_comp['Metric'].str.upper()

    sns.barplot(data=melted_comp, x='Strategy', y='Score', hue='Metric', palette='viridis', errorbar=None)
    plt.title("Vergleich der Retrieval-Leistung (NDCG, RECALL, PRECISION, MAP, MRR @10)\nzwischen Fixed, Recursive, Sentence und Semantic Chunking")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("Strategie")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/13_Vergleich_Retrieval_Leistung.png")
    plt.close()

# 14. Correlation Chunking Time vs Performance
if 'chunking_time_h' in df.columns and 'ndcg@10' in df.columns:
    agg_chunk = df.groupby(['experiment', 'Strategy'])[['chunking_time_h', 'ndcg@10']].mean().reset_index()
    # Filter out 0 or NaN times for log scale safety
    agg_chunk = agg_chunk[agg_chunk['chunking_time_h'] > 0]

    if not agg_chunk.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=agg_chunk, x='chunking_time_h', y='ndcg@10', hue='Strategy', palette=palette, s=100, alpha=0.7)
        plt.title("Korrelation: Chunking-Zeit vs. Performance (NDCG@10)")
        plt.xlabel("Chunking Zeit (Stunden)")
        plt.ylabel("NDCG@10")

        plt.grid(True, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{PLOT_FOLDER}/14_Correlation_Chunking_Time.png")
        plt.close()

# 15. Correlation Indexing Time vs Performance
if 'indexing_time_h' in df.columns and 'ndcg@10' in df.columns:
    agg_index = df.groupby(['experiment', 'Strategy'])[['indexing_time_h', 'ndcg@10']].mean().reset_index()
    agg_index = agg_index[agg_index['indexing_time_h'] > 0]

    if not agg_index.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=agg_index, x='indexing_time_h', y='ndcg@10', hue='Strategy', palette=palette, s=100, alpha=0.7)
        plt.title("Korrelation: Indexierungs-Zeit vs. Performance (NDCG@10)")
        plt.xlabel("Indexierungs-Zeit (Stunden)")
        plt.ylabel("NDCG@10")

        plt.grid(True, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{PLOT_FOLDER}/15_Correlation_Indexing_Time.png")
        plt.close()

# 16. Correlation Total Time (Chunking + Indexing) vs Performance
if 'total_prep_time_h' in df.columns and 'ndcg@10' in df.columns:
    agg_total = df.groupby(['experiment', 'Strategy'])[['total_prep_time_h', 'ndcg@10']].mean().reset_index()
    agg_total = agg_total[agg_total['total_prep_time_h'] > 0]

    if not agg_total.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=agg_total, x='total_prep_time_h', y='ndcg@10', hue='Strategy', palette=palette, s=100, alpha=0.7)
        plt.title("Korrelation: Gesamtzeit (Chunking + Indexierung) vs. Performance (NDCG@10)")
        plt.xlabel("Gesamte Vorbereitungszeit (Stunden)")
        plt.ylabel("NDCG@10")

        plt.grid(True, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{PLOT_FOLDER}/16_Correlation_Total_Time.png")
        plt.close()

# 17. Vector Count Strategy
if 'chunk_total_count' in df.columns:
    plt.figure(figsize=(10, 6))
    # Aggregate to get unique counts per experiment, not per query
    # Assuming chunk_total_count is constant per experiment
    vec_counts = df.groupby(['experiment', 'Strategy'])['chunk_total_count'].mean().reset_index()

    sns.barplot(data=vec_counts, x='Strategy', y='chunk_total_count', palette=palette, errorbar='sd')
    plt.title("Abbildung 5: Anzahl der generierten Vektoren im Vergleich zur Strategie")
    plt.ylabel("Anzahl Vektoren (Total)")
    plt.xlabel("Strategie")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/17_Vector_Count_Strategy.png")
    plt.close()

print(f"✅ ALL DONE. Report and 17 Plots saved in '{PLOT_FOLDER}'.")
