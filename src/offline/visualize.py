import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
CSV_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

# Aesthetic Colors
COLORS = {
    "Faster-Whisper": "#3498db", 
    "WhisperX": "#e74c3c", 
    "whisper.cpp": "#2ecc71"
}

sns.set_theme(style="whitegrid", context="talk")

def save_plot(filename):
    """Saves the current plot to the graph directory."""
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def generate_visualizations():
    if not os.path.exists(CSV_FILE):
        print(f"Results file not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Pre-process percentage columns
    df["WER %"] = df["WER"] * 100
    df["CER %"] = df["CER"] * 100
    
    # 1. RTF vs Duration Line Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="RTF", hue="Model", palette=COLORS, marker="o", style="Model")
    plt.axhline(y=1.0, color='r', linestyle='--', label='Real-time Limit (1.0)')
    plt.xlabel("Audio Duration (s)")
    plt.ylabel("Real Time Factor (RTF)")
    plt.title("RTF vs Audio Duration")
    plt.legend()
    save_plot("rtf_vs_duration.png")

    # 2. Latency vs Duration Line Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="Latency", hue="Model", palette=COLORS, marker="o", style="Model")
    plt.xlabel("Audio Duration (s)")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs Audio Duration")
    save_plot("latency_vs_duration.png")

    # Helper for Bar Plots
    def create_bar_plot(y_col, filename, y_label, title):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x="Model", y=y_col, palette=COLORS, errorbar=None)
        plt.ylabel(y_label)
        plt.title(title)
        ax = plt.gca()
        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', padding=3)
        sns.despine(left=True)
        save_plot(filename)

    create_bar_plot("RTF", "rtf_overall.png", "RTF", "Average Real Time Factor")
    create_bar_plot("WER %", "wer_overall.png", "WER (%)", "Average Word Error Rate")
    create_bar_plot("CER %", "cer_overall.png", "CER (%)", "Average Character Error Rate")
    create_bar_plot("LoadTime", "load_time.png", "Load Time (s)", "Model Load Time")
    
    # 3. Memory Usage
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="PeakMemory", palette=COLORS, errorbar=None)
    plt.ylabel("Memory (MB)")
    plt.title("Peak Memory Usage")
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt='%.0f MB', padding=3)
    save_plot("memory_usage.png")

    # 4. Usage Summary Table
    summary = df.groupby("Model")[["WER", "CER", "RTF", "Latency", "LoadTime", "PeakMemory"]].mean()
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"))
    
    # Create Table Image
    table_data = summary[["WER", "CER", "RTF", "LoadTime", "PeakMemory"]].copy()
    table_data["WER"] = (table_data["WER"] * 100).map('{:.2f}%'.format)
    table_data["CER"] = (table_data["CER"] * 100).map('{:.2f}%'.format)
    table_data["RTF"] = table_data["RTF"].map('{:.3f}'.format)
    table_data["PeakMemory"] = table_data["PeakMemory"].map('{:.0f} MB'.format)
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    tbl = plt.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2)
    plt.title("Performance Summary")
    save_plot("performance_summary_table.png")

    print("\nBenchmark Summary:")
    print(summary)

if __name__ == "__main__":
    generate_visualizations()
