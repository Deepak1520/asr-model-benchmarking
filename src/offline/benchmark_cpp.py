import os
import time
import json
import glob
import re
import threading
import subprocess
import psutil
import pandas as pd
import jiwer
import logging

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up from src/offline/ to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions", "whisper.cpp")

# Whisper.cpp paths
WHISPER_CPP_DIR = os.path.join(PROJECT_ROOT, "whisper.cpp")
MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/ggml-large-v2.bin")
BINARY_PATH = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")

if not os.path.exists(BINARY_PATH):
    # Try fallback name
    BINARY_PATH = BINARY_PATH.replace("whisper-cli", "main")
    if not os.path.exists(BINARY_PATH):
        raise FileNotFoundError(f"whisper.cpp binary not found at {BINARY_PATH}. Please clean build whisper.cpp.")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage(pid: int) -> float:
    """Returns the RSS memory usage of the process in MB."""
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

def run_benchmark():
    logger.info("Starting whisper.cpp benchmark...")

    if not os.path.exists(os.path.join(DATA_DIR, "ground_truth.json")):
        logger.error("Ground truth file not found.")
        return

    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.wav"), recursive=True))
    files = [f for f in files if "temp" not in f]

    results = []

    for wav_file in files:
        fname = os.path.basename(wav_file)
        if fname not in ground_truth:
            continue

        # Determine Category
        category = "Unknown"
        if "clean" in fname: category = "Clean"
        elif "noisy" in fname: category = "Noisy"
        elif "accented" in fname: category = "Accented"

        logger.info(f"Processing {fname} ({category})...")

        cmd = [
            BINARY_PATH,
            "-m", MODEL_PATH,
            "-f", wav_file,
            "--no-timestamps"
        ]

        start_time = time.time()
        
        # Background memory tracking
        peak_memory = 0.0
        stop_tracking = False
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        def track_memory():
            nonlocal peak_memory
            while not stop_tracking:
                if process.poll() is not None:
                    break
                peak_memory = max(peak_memory, get_memory_usage(process.pid))
                time.sleep(0.1)

        tracker_thread = threading.Thread(target=track_memory)
        tracker_thread.start()

        stdout, stderr = process.communicate()
        stop_tracking = True
        tracker_thread.join()

        latency = time.time() - start_time

        # Parse load time from stderr (look for "load time = 123.45 ms")
        load_time = 0.0
        match = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", stderr)
        if match:
            load_time = float(match.group(1)) / 1000.0

        # Duration calculation (16kHz 16-bit mono = 32000 bytes/sec)
        duration = os.path.getsize(wav_file) / 32000.0
        rtf = latency / duration if duration > 0 else 0

        transcript = stdout.strip()
        reference = ground_truth[fname]
        
        wer = jiwer.wer(reference, transcript)
        cer = jiwer.cer(reference, transcript)

        # Save individual transcription
        out_txt_path = os.path.join(TRANSCRIPTIONS_DIR, fname.replace(".wav", ".txt"))
        with open(out_txt_path, "w") as f:
            f.write(f"Reference: {reference}\n\nPrediction: {transcript}\n\nWER: {wer}")

        results.append({
            "Model": "whisper.cpp",
            "Filename": fname,
            "Category": category,
            "Duration": duration,
            "Latency": latency,
            "RTF": rtf,
            "WER": wer,
            "CER": cer,
            "LoadTime": load_time,
            "PeakMemory": peak_memory
        })

        logger.info(f"  > Latency: {latency:.2f}s | RTF: {rtf:.3f} | MEM: {peak_memory:.1f}MB | WER: {wer:.3f}")
        time.sleep(1) # Cool down

    # Save aggregated results
    new_df = pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_csv(RESULTS_FILE)
        # Remove previous runs for this model to avoid duplicates
        existing_df = existing_df[existing_df["Model"] != "whisper.cpp"]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(RESULTS_FILE, index=False)
    else:
        new_df.to_csv(RESULTS_FILE, index=False)

    logger.info(f"Benchmark completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()
