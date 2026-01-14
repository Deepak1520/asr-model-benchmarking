import os
import time
import json
import glob
import threading
import psutil
import pandas as pd
import jiwer
from faster_whisper import WhisperModel

# config
MODEL_SIZE = "large-v2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_FILE = os.path.join(DATA_DIR, "benchmark_results.csv")

def get_memory_usage():
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
            except: pass
            time.sleep(0.1)

def run_benchmark():
    print("Running Faster-Whisper benchmark...")
    
    # setup paths
    dataset_dir = os.path.join(DATA_DIR, "dataset")
    transcriptions_dir = os.path.join(DATA_DIR, "transcriptions", "Faster-Whisper")
    os.makedirs(transcriptions_dir, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    # load model
    curr_mem = MemoryMonitor()
    curr_mem.start()
    
    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    files = glob.glob(os.path.join(dataset_dir, "**/*.wav"), recursive=True)
    files = sorted([f for f in files if "temp" not in f])
    
    results = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        if fname not in ground_truth: continue
        
        # determine category
        cat = "Unknown"
        if "clean" in fpath: cat = "Clean"
        elif "noisy" in fpath: cat = "Noisy"
        elif "accented" in fpath: cat = "Accented"
            
        print(f"Transcribing {fname}...")
        
        start = time.time()
        segments, info = model.transcribe(fpath, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        latency = time.time() - start
        
        duration = info.duration
        rtf = latency / duration if duration > 0 else 0
        
        # calc metrics
        ref = ground_truth[fname]
        wer = jiwer.wer(ref, text)
        cer = jiwer.cer(ref, text)
        
        # save txt
        with open(os.path.join(transcriptions_dir, fname.replace(".wav", ".txt")), "w") as f:
            f.write(f"Ref: {ref}\nPred: {text}\nWER: {wer}\nCER: {cer}")
            
        results.append({
            "Model": "Faster-Whisper",
            "Filename": fname,
            "Category": cat,
            "Duration": duration,
            "Latency": latency,
            "RTF": rtf,
            "WER": wer,
            "CER": cer,
            "LoadTime": load_time,
            "PeakMemory": 0 
        })
        
        print(f"  -> Latency: {latency:.2f}s, RTF: {rtf:.2f}, WER: {wer:.2f}")
        time.sleep(2) # cool down

    peak_mem = curr_mem.stop()
    print(f"Peak Memory: {peak_mem:.1f} MB")
    
    # update results
    for r in results: r["PeakMemory"] = peak_mem
    
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        old_df = pd.read_csv(RESULTS_FILE)
        # remove old runs
        old_df = old_df[old_df["Model"] != "Faster-Whisper"]
        df = pd.concat([old_df, df])
        
    df.to_csv(RESULTS_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    run_benchmark()
