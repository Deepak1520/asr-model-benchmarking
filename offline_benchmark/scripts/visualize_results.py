import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE, "data", "benchmark_results.csv")
GRAPH_DIR = os.path.join(BASE, "results", "graphs")
SUMMARY_DIR = os.path.join(BASE, "results")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# nice colors
COLORS = {
    "Faster-Whisper": "#3498db", 
    "WhisperX": "#e74c3c", 
    "whisper.cpp": "#2ecc71"
}

sns.set_theme(style="whitegrid", context="talk")

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, name), dpi=300)
    plt.close()
    print(f"Saved {name}")

def make_graphs():
    if not os.path.exists(CSV_FILE):
        print("No results found.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # % scaling
    df["WER %"] = df["WER"] * 100
    df["CER %"] = df["CER"] * 100
    
    # RTF vs Duration
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="RTF", hue="Model", palette=COLORS, marker="o", style="Model")
    plt.axhline(y=1.0, color='r', linestyle='--', label='Real-time (1.0)')
    plt.xlabel("Audio Duration (s)")
    plt.ylabel("RTF")
    plt.legend()
    save("rtf_vs_duration.png")

    # Latency vs Duration
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="Latency", hue="Model", palette=COLORS, marker="o", style="Model")
    plt.xlabel("Audio Duration (s)")
    plt.ylabel("Latency (s)")
    save("latency_vs_duration.png")

    # Bar plots helper
    def bar_plot(y_col, fname, label):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x="Model", y=y_col, palette=COLORS, errorbar=None)
        plt.ylabel(label)
        ax = plt.gca()
        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', padding=3)
        sns.despine(left=True)
        save(fname)

    bar_plot("RTF", "rtf_overall.png", "RTF")
    bar_plot("WER %", "wer_overall.png", "WER (%)")
    bar_plot("CER %", "cer_overall.png", "CER (%)")
    bar_plot("LoadTime", "load_time.png", "Load Time (s)")
    
    # Memory
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="PeakMemory", palette=COLORS, errorbar=None)
    plt.ylabel("Memory (MB)")
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt='%.0f MB', padding=3)
    save("memory_usage.png")

    # Summary CSV
    summary = df.groupby("Model")[["WER", "CER", "RTF", "Latency", "LoadTime", "PeakMemory"]].mean()
    summary.to_csv(os.path.join(SUMMARY_DIR, "summary_results.csv"))
    print("\nResults:")
    print(summary)

    # Summary Table Image
    sum_table = summary[["WER", "CER", "RTF", "LoadTime", "PeakMemory"]].copy()
    sum_table["WER"] = (sum_table["WER"] * 100).map('{:.2f}%'.format)
    sum_table["CER"] = (sum_table["CER"] * 100).map('{:.2f}%'.format)
    sum_table["RTF"] = sum_table["RTF"].map('{:.3f}'.format)
    sum_table["PeakMemory"] = sum_table["PeakMemory"].map('{:.0f} MB'.format)
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    tbl = plt.table(cellText=sum_table.values, colLabels=sum_table.columns, rowLabels=sum_table.index, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2)
    save("performance_summary_table.png")

if __name__ == "__main__":
    make_graphs()
