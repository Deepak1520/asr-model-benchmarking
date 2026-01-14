import os
import time
import json
import glob
import threading
import psutil
import pandas as pd
import jiwer
import whisperx
import torch

# pytorch 2.6 fix
try:
    import torch.serialization
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig
    ])
except: pass

MODEL_SIZE = "large-v2"
DEVICE = "cpu" # MPS sucks for this on some versions, sticking to CPU
COMPUTE_TYPE = "int8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_FILE = os.path.join(DATA_DIR, "benchmark_results.csv")

class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.t = threading.Thread(target=self.loop)
    
    def start(self):
        self.t.start()
        
    def stop_mon(self):
        self.stop.set()
        self.t.join()
        return self.peak
        
    def loop(self):
        p = psutil.Process()
        while not self.stop.is_set():
            m = p.memory_info().rss / 1024**2
            self.peak = max(self.peak, m)
            time.sleep(0.1)

def main():
    print("Starting WhisperX benchmark...")
    
    mem = MemoryMonitor()
    mem.start()
    
    t0 = time.time()
    try:
        model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    load_time = time.time() - t0
    print(f"Model loaded: {load_time:.2f}s")
    
    dataset_dir = os.path.join(DATA_DIR, "dataset")
    out_dir = os.path.join(DATA_DIR, "transcriptions", "WhisperX")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        gt = json.load(f)
        
    files = glob.glob(os.path.join(dataset_dir, "**/*.wav"), recursive=True)
    files = sorted([f for f in files if "temp" not in f])
    
    results = []
    
    for path in files:
        fname = os.path.basename(path)
        if fname not in gt: continue
        
        cat = "Clean" if "clean" in path else "Noisy" if "noisy" in path else "Accented"
        print(f"Processing {fname}...")
        
        # transcribe
        t_start = time.time()
        audio = whisperx.load_audio(path)
        res = model.transcribe(audio, batch_size=4)
        text = " ".join([x["text"] for x in res["segments"]]).strip()
        latency = time.time() - t_start
        
        dur = len(audio) / 16000
        rtf = latency / dur
        
        # score
        ref = gt[fname]
        wer = jiwer.wer(ref, text)
        cer = jiwer.cer(ref, text)
        
        # save transcript
        with open(os.path.join(out_dir, fname.replace(".wav", ".txt")), "w") as f:
            f.write(f"REF: {ref}\nHYP: {text}\nWER: {wer}")
            
        results.append({
            "Model": "WhisperX",
            "Filename": fname,
            "Category": cat,
            "Duration": dur,
            "Latency": latency,
            "RTF": rtf,
            "WER": wer,
            "CER": cer,
            "LoadTime": load_time
        })
        
        print(f"  Result: {latency:.2f}s (RTF {rtf:.2f}), WER {wer:.2f}")
        time.sleep(2)
        
    peak = mem.stop_mon()
    print(f"Peak mem: {peak:.2f} MB")
    
    for r in results: r["PeakMemory"] = peak
    
    # save csv
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        prev = pd.read_csv(RESULTS_FILE)
        prev = prev[prev["Model"] != "WhisperX"]
        df = pd.concat([prev, df])
        
    df.to_csv(RESULTS_FILE, index=False)
    print("Saved results.")

if __name__ == "__main__":
    main()
