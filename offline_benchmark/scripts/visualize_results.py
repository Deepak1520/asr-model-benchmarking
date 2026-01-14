import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust paths for offline_benchmark execution context
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "data", "benchmark_results.csv")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")
SUMMARY_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Colors
MODEL_COLORS = {
    "Faster-Whisper": "#3498db",  # Blue
    "WhisperX": "#e74c3c",         # Red
    "whisper.cpp": "#2ecc71"       # Green
}

sns.set_theme(style="whitegrid", context="talk")

def save_plot(filename):
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def generate_visuals():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file {RESULTS_FILE} not found.")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    # Convert to percentages for better visualization
    df["WER %"] = df["WER"] * 100
    df["CER %"] = df["CER"] * 100
    
    # 1. RTF vs Duration (Scatter/Line)
    # This is the most important plot for "varied length" analysis
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="RTF", hue="Model", palette=MODEL_COLORS, marker="o", style="Model")
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time (1.0)')
    plt.xlabel("Audio Duration (seconds)", fontsize=14)
    plt.ylabel("RTF (Lower is Better)", fontsize=14)
    plt.legend()
    sns.despine()
    save_plot("rtf_vs_duration.png")

    # 2. Latency vs Duration
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Duration", y="Latency", hue="Model", palette=MODEL_COLORS, marker="o", style="Model")
    plt.xlabel("Audio Duration (seconds)", fontsize=14)
    plt.ylabel("Latency (seconds)", fontsize=14)
    plt.legend()
    sns.despine()
    save_plot("latency_vs_duration.png")
    save_plot("latency_vs_duration.png")

    # 2a. Average Latency (1-minute samples only)
    plt.figure(figsize=(8, 6))
    # Filter for samples approx 1 minute (e.g., < 70s to allow for variance)
    df_1min = df[df["Duration"] < 80]
    sns.barplot(data=df_1min, x="Model", y="Latency", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("Latency (seconds)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("latency_1min.png")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="Latency", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("Latency (seconds)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("latency_avg.png")

    # 2c. Average RTF (Bar Chart) - Requested Overall
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="RTF", palette=MODEL_COLORS, errorbar=None)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    plt.text(x=1.0, y=1.0, s="Real-Time Limit (1.0)", color='red', fontsize=12, va='center', ha='center', backgroundcolor='white')
    plt.ylabel("RTF (Lower is Better)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("rtf_overall.png")

    # 2d. Average WER (Bar Chart) - Requested Overall
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="WER %", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("WER (%) (Lower is Better)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("wer_overall.png")

    # 2e. Average CER (Bar Chart) - Requested Overall
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="CER %", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("CER (%) (Lower is Better)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("cer_overall.png")



    # 4. Memory Usage (Boxplot or Bar)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="PeakMemory", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("Memory (MB)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f MB', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("memory_usage.png")
    
    # 5. Load Time (Bar)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="LoadTime", palette=MODEL_COLORS, errorbar=None)
    plt.ylabel("Time (seconds)", fontsize=14)
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs', fontsize=12, padding=3)
    sns.despine(left=True)
    save_plot("load_time.png")

    # Summary Text
    summary = df.groupby("Model")[["WER", "CER", "RTF", "Latency", "LoadTime", "PeakMemory"]].mean()
    summary.to_csv(os.path.join(SUMMARY_DIR, "summary_results.csv"))
    print("\nSummary Results:")
    print(summary)

    # 6. Generate Summary Table Image
    # Select columns: WER, CER, RTF, LoadTime, PeakMemory (Latency removed per request)
    summary_table = summary[["WER", "CER", "RTF", "LoadTime", "PeakMemory"]].copy()
    
    # Format columns for display
    summary_table["WER"] = (summary_table["WER"] * 100).map('{:.2f}%'.format)
    summary_table["CER"] = (summary_table["CER"] * 100).map('{:.2f}%'.format)
    summary_table["RTF"] = summary_table["RTF"].map('{:.3f}'.format)
    summary_table["LoadTime"] = summary_table["LoadTime"].map('{:.2f}s'.format)
    summary_table["PeakMemory"] = summary_table["PeakMemory"].map('{:.0f} MB'.format)
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    table = plt.table(cellText=summary_table.values,
                      colLabels=["WER", "CER", "RTF", "Load Time", "Peak Memory"],
                      rowLabels=summary_table.index,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    save_plot("performance_summary_table.png")

if __name__ == "__main__":
    generate_visuals()

