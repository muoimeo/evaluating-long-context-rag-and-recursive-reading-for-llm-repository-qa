import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(results_dir="results"):
    """Load all evaluation JSON files into a single Pandas DataFrame."""
    all_records = []
    
    # Map filenames to method names for better display
    method_map = {
        "eval_method_a.json": "Method A (Long Context)",
        "eval_method_b.json": "Method B (RAG)",
        "eval_method_c.json": "Method C (Recursive)"
    }
    
    fallback_map = {"A": "Method A (Long Context)", "B": "Method B (RAG)", "C": "Method C (Recursive)"}
    
    for file_path in glob.glob(os.path.join(results_dir, 'eval_method_*.json')):
        filename = os.path.basename(file_path)
        method_name = method_map.get(filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Invalid JSON")
                continue
                
        for item in data:
            if not method_name:
                 method_name = fallback_map.get(item.get("method", "Unknown"), "Unknown")
                 
            record = {
                "Method": method_name,
                "Question ID": item.get("id"),
                "Dataset": item.get("dataset_type", "Unknown").replace("_", " ").title(),
                "Difficulty": item.get("difficulty", "Unknown").title(),
                "Success": item.get("success", False),
                "Accuracy": item.get("accuracy_score", 0.0),
                "Citation": item.get("citation_score", 0.0),
                "Latency (s)": item.get("latency_sec", 0.0),
                "Input Tokens": item.get("input_tokens", 0),
                "Output Tokens": item.get("output_tokens", 0),
                "Total Tokens": item.get("input_tokens", 0) + item.get("output_tokens", 0)
            }
            all_records.append(record)
            
    return pd.DataFrame(all_records)

def create_plots(df, output_dir="results/plots"):
    """Generate and save various comparative plots."""
    if df.empty:
        print("No data found to plot. Ensure eval_method_*.json files exist in results/")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    
    # Colors for consistent method representation
    palette = sns.color_palette("Set2", n_colors=df['Method'].nunique())
    
    print(f"Loaded {len(df)} evaluation records. Generating plots...")

    # 1. Bar Chart: Average Accuracy and Citation by Method (Multi-graph)
    plt.figure(figsize=(12, 6))
    melted_scores = pd.melt(df, id_vars=['Method'], value_vars=['Accuracy', 'Citation'], 
                            var_name='Metric', value_name='Score')
    ax = sns.barplot(data=melted_scores, x='Method', y='Score', hue='Metric', errorbar=None, palette="viridis")
    plt.title('Average Accuracy and Citation Scores by Method', pad=20)
    plt.ylim(0, 1.0)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_avg_scores_comparison.png'), dpi=300)
    plt.close()

    # 2. Boxplot: Latency Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Method', y='Latency (s)', palette=palette)
    sns.stripplot(data=df, x='Method', y='Latency (s)', color=".25", size=5, alpha=0.6)
    plt.title('Distribution of Response Latency', pad=20)
    plt.ylabel('Latency (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_latency_distribution.png'), dpi=300)
    plt.close()

    # 3. Bar Chart: Average Token Usage
    plt.figure(figsize=(12, 6))
    melted_tokens = pd.melt(df, id_vars=['Method'], value_vars=['Input Tokens', 'Output Tokens'], 
                            var_name='Token Type', value_name='Count')
    ax = sns.barplot(data=melted_tokens, x='Method', y='Count', hue='Token Type', errorbar=None, palette="magma")
    plt.title('Average Token Usage by Method', pad=20)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.0f', padding=3)
    plt.ylabel('Number of Tokens')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_token_usage_comparison.png'), dpi=300)
    plt.close()

    # 4. Scatter Plot: Latency vs. Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Latency (s)', y='Accuracy', hue='Method', style='Dataset', 
                    s=150, alpha=0.8, palette=palette)
    plt.title('Latency vs. Accuracy Trade-off', pad=20)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_latency_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Grouped Bar: Accuracy by Dataset Type
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Method', errorbar=None, palette=palette)
    plt.title('Accuracy Score by Dataset Category', pad=20)
    plt.ylim(0, 1.0)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_accuracy_by_dataset.png'), dpi=300)
    plt.close()

    # 6. Grouped Bar: Citation by Dataset Type
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Dataset', y='Citation', hue='Method', errorbar=None, palette=palette)
    plt.title('Citation Score by Dataset Category', pad=20)
    plt.ylim(0, 1.0)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_citation_by_dataset.png'), dpi=300)
    plt.close()
    
    # 7. Scatter/Bubble: Accuracy vs Citation (Size = Latency)
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(data=df, x='Accuracy', y='Citation', hue='Method', size='Latency (s)',
                              sizes=(50, 500), alpha=0.7, palette=palette)
    plt.title('Accuracy vs Citation (Bubble size = Latency)', pad=20)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_accuracy_vs_citation_bubble.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All plots successfully saved to {os.path.abspath(output_dir)}")
    print("Files created:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")

if __name__ == "__main__":
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Missing required visualization libraries.")
        print("Please install them using: pip install pandas matplotlib seaborn")
        exit(1)
        
    df = load_data()
    create_plots(df)
