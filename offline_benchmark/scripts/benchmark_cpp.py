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

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)

MODEL = os.path.join(ROOT, "whisper.cpp/models/ggml-large-v2.bin")
BINARY = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-cli")

if not os.path.exists(BINARY):
    BINARY = BINARY.replace("whisper-cli", "main") # try old name
    if not os.path.exists(BINARY):
        print(f"ERROR: whisper.cpp binary not found at {BINARY}")
        exit(1)

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS = os.path.join(DATA_DIR, "benchmark_results.csv")

def get_mem(pid):
    try:
        return psutil.Process(pid).memory_info().rss / 1024**2
    except: return 0

def run_cpp_benchmark():
    print("Benchmarking whisper.cpp...")
    
    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)
        
    dataset_dir = os.path.join(DATA_DIR, "dataset")
    out_dir = os.path.join(DATA_DIR, "transcriptions", "whisper.cpp")
    os.makedirs(out_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.wav"), recursive=True))
    files = [f for f in files if "temp" not in f]

    results = []
    
    for wav_file in files:
        fname = os.path.basename(wav_file)
        if fname not in ground_truth: continue
        
        cat = "Unknown"
        if "clean" in fname: cat = "Clean"
        elif "noisy" in fname: cat = "Noisy"
        elif "accented" in fname: cat = "Accented"
        
        print(f"Running on {fname} ({cat})...")
        
        cmd = [
            BINARY, 
            "-m", MODEL,
            "-f", wav_file,
            "--no-timestamps"
        ]
        
        start = time.time()
        
        # track memory in bg
        peak_mem = 0
        stop_mem = False
        
        def track():
            nonlocal peak_mem
            while not stop_mem:
                if p.poll() is not None: break
                try: peak_mem = max(peak_mem, get_mem(p.pid))
                except: pass
                time.sleep(0.1)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t = threading.Thread(target=track)
        t.start()
        
        out, err = p.communicate()
        stop_mem = True
        t.join()
        
        latency = time.time() - start
        
        # parse stderr for load time
        # look for "load time =   123.45 ms"
        load_time = 0
        m = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", err)
        if m: load_time = float(m.group(1)) / 1000
        
        duration = os.path.getsize(wav_file) / 32000 # 16khz 16bit mono = 32000 bytes/sec
        rtf = latency / duration
        
        transcript = out.strip()
        ref = ground_truth[fname]
        wer = jiwer.wer(ref, transcript)
        cer = jiwer.cer(ref, transcript)
        
        # save
        with open(os.path.join(out_dir, fname.replace(".wav", ".txt")), "w") as f:
            f.write(f"{ref}\n\n{transcript}\n\nWER: {wer}")

        results.append({
            "Model": "whisper.cpp",
            "Filename": fname,
            "Category": cat,
            "Duration": duration,
            "Latency": latency,
            "RTF": rtf,
            "WER": wer,
            "CER": cer,
            "LoadTime": load_time,
            "PeakMemory": peak_mem
        })
        
        print(f"  > Latency: {latency:.2f}s, RTF: {rtf:.3f}, MEM: {peak_mem:.0f}MB")
        time.sleep(1)

    # save results
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS):
        prev = pd.read_csv(RESULTS)
        prev = prev[prev["Model"] != "whisper.cpp"]
        df = pd.concat([prev, df], ignore_index=True)
        
    df.to_csv(RESULTS, index=False)
    print("Saved benchmark results.")

if __name__ == "__main__":
    run_cpp_benchmark()
