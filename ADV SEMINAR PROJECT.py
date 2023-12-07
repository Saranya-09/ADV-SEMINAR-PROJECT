import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from gprofiler import GProfiler

# Load Data
file_path = r"C:\Users\sunny\Downloads\GSE226687_RNA-seq_AllNormalizedCounts_PeakNorm.txt.gz"
counts_df = pd.read_csv(file_path, index_col=0, compression='gzip', sep='\t')
norm_counts = np.log2(counts_df + 1)

# Identify Genes with Low Variance
threshold = 0.1
low_var_genes = norm_counts.index[norm_counts.var(axis=1) < threshold]
norm_counts_filtered = norm_counts.drop(low_var_genes)

# Split Conditions
control = norm_counts_filtered[norm_counts_filtered.columns[:3]]
treated = norm_counts_filtered[norm_counts_filtered.columns[3:]]

# Perform t-test
pvalues = [ttest_ind(control.loc[gene], treated.loc[gene], nan_policy='omit')[1] for gene in norm_counts_filtered.index]

# Perform Multiple Testing Correction
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

# Create DataFrame with Results
results_df = pd.DataFrame({
    'gene': norm_counts_filtered.index,
    'log2FoldChange': np.mean(treated, axis=1) - np.mean(control, axis=1),
    'padj': pvals_corrected
})

# Get Significant Genes
sig_genes = results_df[results_df['padj'] < 0.05]['gene']

# Functional Enrichment Analysis
gp = GProfiler(return_dataframe=True)
enrichment_results = gp.profile(organism='hsapiens', query=sig_genes.tolist(), sources=['GO:BP', 'KEGG'], user_threshold=0.05)

# Print Enrichment Results
print(enrichment_results[['native', 'name', 'p_value', 'intersection_size']])

# Volcano Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log2FoldChange', y=-np.log10(results_df['padj']), data=results_df, hue=(results_df['padj'] < 0.05))
plt.title('Volcano Plot')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-log10(Adjusted P-value)')
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
sns.clustermap(np.log2(norm_counts_filtered + 1), cmap='viridis', method='average', col_cluster=False)
plt.title('Heatmap of Normalized Counts')
plt.show()

# Venn Diagram
alpha = 0.05

# Identify upregulated genes
upregulated_genes = results_df[(results_df['log2FC'] > 0) & (results_df['AdjPval'] < alpha)]['GeneID'].tolist()

# Identify downregulated genes
downregulated_genes = results_df[(results_df['log2FC'] < 0) & (results_df['AdjPval'] < alpha)]['GeneID'].tolist()

# Print identified genes
print("Upregulated Genes:")
print(upregulated_genes)
print("\nDownregulated Genes:")
print(downregulated_genes)

# Create a Venn diagram
venn_labels = {'100': len(upregulated_genes) - len(set(upregulated_genes).intersection(downregulated_genes)),
               '010': len(downregulated_genes) - len(set(upregulated_genes).intersection(downregulated_genes)),
               '110': len(set(upregulated_genes).intersection(downregulated_genes))}

# Plot the Venn diagram
venn2(subsets=(len(upregulated_genes), len(downregulated_genes), len(set(upregulated_genes).intersection(downregulated_genes))),
      set_labels=('Upregulated Genes', 'Downregulated Genes'))

# Display the intersection size in the center
for idx, label in enumerate(venn_labels):
    plt.text(*venn2.get_label_by_id(label).get_position(), str(venn_labels[label]), ha='center', va='center', fontsize=10)

plt.title('Venn Diagram of Upregulated and Downregulated Genes')
plt.show()

# Get the top 10 DEGs
top_n_genes = results_df.nlargest(10, 'log2FC')

# Create a bar plot for the top 10 DEGs
plt.figure(figsize=(10, 6))
sns.barplot(x='log2FC', y='GeneID', data=top_n_genes)
plt.title('Top 10 Differentially Expressed Genes')
plt.xlabel('Log2FC')
plt.ylabel('Gene Name')
plt.show()