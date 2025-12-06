import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_FILE = "benchmark_results.csv"
OUTPUT_IMAGE = "benchmark_analysis.png"

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

def visualize():
    if not os.path.exists(CSV_FILE):
        print(f"Error: File '{CSV_FILE}' not found.")
        return

    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print("Error: The CSV file is empty.")
        return

    df['Query_ID'] = df['Query_ID'].astype(str)
    num_iterations = df['Iteration'].max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Benchmark Analysis: Stateless vs. Session Mode vs. Persistent (N={num_iterations})', 
                 fontsize=16, fontweight='bold', y=1.02)

    sns.barplot(data=df, x="Query_ID", y="Input_Tokens", hue="Mode",
                palette="viridis", errorbar='sd', capsize=.1, ax=axes[0],
                legend=True)
    axes[0].set_title("Input Tokens (Cost Impact)\n(Mean ± Std Dev)")
    axes[0].set_ylabel("Token Count")
    axes[0].set_xlabel("Query Sequence (Q1-Q5)")

    sns.barplot(data=df, x="Query_ID", y="Output_Tokens", hue="Mode",
                palette="magma", errorbar='sd', capsize=.1, ax=axes[1],
                legend=True)
    axes[1].set_title("Output Tokens (Response Volume)\n(Mean ± Std Dev)")
    axes[1].set_ylabel("Token Count")
    axes[1].set_xlabel("Query Sequence (Q1-Q5)")

    sns.barplot(data=df, x="Query_ID", y="Latency(s)", hue="Mode",
                palette="rocket", errorbar='sd', capsize=.1, ax=axes[2],
                legend=True)
    axes[2].set_title("End-to-End Latency\n(Mean ± Std Dev)")
    axes[2].set_ylabel("Time (seconds)")
    axes[2].set_xlabel("Query Sequence (Q1-Q5)")

    for ax in axes:
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', 
                  borderaxespad=0, frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.subplots_adjust(right=0.85)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {os.path.abspath(OUTPUT_IMAGE)}")
    plt.close()

if __name__ == "__main__":
    visualize()