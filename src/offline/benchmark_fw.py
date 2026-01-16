import os
import time
import json
import glob
import threading
import psutil
import pandas as pd
import jiwer
import logging
from faster_whisper import WhisperModel

# --- Configuration ---
MODEL_SIZE = "large-v2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions", "Faster-Whisper")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Returns total RSS memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

class MemoryMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self._run)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.peak

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.peak = max(self.peak, get_memory_usage())
            except Exception:
                pass
            time.sleep(0.1)

def run_benchmark():
    logger.info("Starting Faster-Whisper benchmark...")
    
    if not os.path.exists(os.path.join(DATA_DIR, "ground_truth.json")):
        logger.error("Ground truth file not found.")
        return

    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    # Load Model
    monitor = MemoryMonitor()
    monitor.start()
    
    t0 = time.time()
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.wav"), recursive=True))
    files = [f for f in files if "temp" not in f]
    
    results = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        if fname not in ground_truth:
            continue
        
        # Determine Category
        category = "Unknown"
        if "clean" in fpath: category = "Clean"
        elif "noisy" in fpath: category = "Noisy"
        elif "accented" in fpath: category = "Accented"
            
        logger.info(f"Transcribing {fname} ({category})...")
        
        start_time = time.time()
        # Using beam_size=5 for fair comparison with standard benchmarks
        segments, info = model.transcribe(fpath, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        latency = time.time() - start_time
        
        duration = info.duration
        rtf = latency / duration if duration > 0 else 0
        
        # Metrics
        ref_text = ground_truth[fname]
        wer = jiwer.wer(ref_text, text)
        cer = jiwer.cer(ref_text, text)
        
        # Save transcription
        with open(os.path.join(TRANSCRIPTIONS_DIR, fname.replace(".wav", ".txt")), "w") as f:
            f.write(f"Ref: {ref_text}\nPred: {text}\nWER: {wer}\nCER: {cer}")
            
        results.append({
            "Model": "Faster-Whisper",
            "Filename": fname,
            "Category": category,
            "Duration": duration,
            "Latency": latency,
            "RTF": rtf,
            "WER": wer,
            "CER": cer,
            "LoadTime": load_time,
            "PeakMemory": 0  # Updated later
        })
        
        logger.info(f"  -> Latency: {latency:.2f}s | RTF: {rtf:.2f} | WER: {wer:.2f}")
        time.sleep(1) # Cool down

    peak_mem = monitor.stop()
    logger.info(f"Peak Memory during benchmark: {peak_mem:.1f} MB")
    
    # Update results with peak memory
    for r in results:
        r["PeakMemory"] = peak_mem
    
    # Save Results
    new_df = pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_csv(RESULTS_FILE)
        existing_df = existing_df[existing_df["Model"] != "Faster-Whisper"]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(RESULTS_FILE, index=False)
    else:
        new_df.to_csv(RESULTS_FILE, index=False)
        
    logger.info("Benchmark completed.")

if __name__ == "__main__":
    run_benchmark()
