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

# --- Configuration ---
MODEL_PATH = "whisper.cpp/models/ggml-large-v2.bin"
BINARY_PATH = "whisper.cpp/build/bin/whisper-cli"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_FILE = os.path.join(BASE_DIR, "data", "benchmark_results.csv")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "transcriptions", "whisper.cpp")

MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)
BINARY_PATH = os.path.join(PROJECT_ROOT, BINARY_PATH)

os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

class MemoryMonitor:
    def __init__(self, target_pid=None, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_memory = 0
        self.target_pid = target_pid
        self.thread = threading.Thread(target=self._monitor)

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.peak_memory

    def _monitor(self):
        try:
            process = psutil.Process(self.target_pid) if self.target_pid else psutil.Process()
        except psutil.NoSuchProcess:
            return

        while not self.stop_event.is_set():
            try:
                mem = process.memory_info().rss / (1024 * 1024) 
                self.peak_memory = max(self.peak_memory, mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except:
                pass
            time.sleep(self.interval)

class BenchmarkCPP:
    def __init__(self):
        self.results = []
        with open(GROUND_TRUTH_FILE, "r") as f:
            self.ground_truth = json.load(f)
            
        if not os.path.exists(BINARY_PATH):
            fallback = BINARY_PATH.replace("whisper-cli", "main")
            if os.path.exists(fallback):
                self.binary_path = fallback
            else:
                raise FileNotFoundError(f"Binary not found at {BINARY_PATH}")
        else:
            self.binary_path = BINARY_PATH

    def get_category(self, path):
        fname = os.path.basename(path).lower()
        if "clean" in fname: return "Clean"
        if "noisy" in fname: return "Noisy"
        if "accented" in fname: return "Accented"
        return "Unknown"

    def run(self):
        print("Benchmarking whisper.cpp...")
        
        files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.wav"), recursive=True))
        files = [f for f in files if "temp" not in f]
        
        print(f"Found {len(files)} files.")

        for audio_path in files:
            fname = os.path.basename(audio_path)
            if fname not in self.ground_truth:
                continue

            category = self.get_category(audio_path)
            print(f"Processing {fname} ({category})...")

            cmd = [
                self.binary_path, 
                "-m", MODEL_PATH, 
                "-f", audio_path,
                "-nt", # No timestamps
                "--no-timestamps"
            ]
            
            start_time = time.time()
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                monitor = MemoryMonitor(target_pid=process.pid)
                monitor.start()
                
                stdout, stderr = process.communicate()
                peak_mem = monitor.stop()
                
                latency = time.time() - start_time
                
                # Parse Load Time from stderr
                load_match = re.search(r"load time\s*=\s*(\d+\.?\d*)\s*ms", stderr)
                load_time = float(load_match.group(1)) / 1000.0 if load_match else 0.0
                
                # Estimate Duration from file size
                duration = os.path.getsize(audio_path) / 32000.0
                rtf = latency / duration if duration > 0 else 0
                
                transcription = stdout.strip()
                
                # Metric
                reference = self.ground_truth[fname]
                wer = jiwer.wer(reference, transcription)
                cer = jiwer.cer(reference, transcription)
                
                self._save_transcription(fname, reference, transcription, wer, cer)
                
                self.results.append({
                    "Model": "whisper.cpp",
                    "Filename": fname,
                    "Category": category,
                    "Duration": duration,
                    "Latency": latency,
                    "RTF": rtf,
                    "WER": wer,
                    "CER": cer,
                    "LoadTime": load_time,
                    "PeakMemory": peak_mem
                })
                
                print(f"  -> Latency: {latency:.2f}s, RTF: {rtf:.2f}, Load: {load_time:.3f}s, Mem: {peak_mem:.2f}MB")
                
            except Exception as e:
                print(f"Error running whisper.cpp: {e}")
            
            time.sleep(5)
            
        self.save_results()

    def _save_transcription(self, fname, ref, trans, wer, cer):
        out_path = os.path.join(TRANSCRIPTIONS_DIR, f"{os.path.splitext(fname)[0]}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Reference:\n{ref}\n\n")
            f.write(f"# Transcription:\n{trans}\n\n")
            f.write(f"# Metrics:\nWER: {wer:.4f} ({wer*100:.2f}%)\nCER: {cer:.4f} ({cer*100:.2f}%)\n")

    def save_results(self):
        new_df = pd.DataFrame(self.results)
        if os.path.exists(RESULTS_FILE):
            try:
                existing_df = pd.read_csv(RESULTS_FILE)
                if "LoadTime" in existing_df.columns:
                    existing_df = existing_df[existing_df["Model"] != "whisper.cpp"]
                else:
                    existing_df = pd.DataFrame()
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            except:
                final_df = new_df
        else:
            final_df = new_df
            
        final_df.to_csv(RESULTS_FILE, index=False)
        print(f"Results updated in {RESULTS_FILE}")

if __name__ == "__main__":
    BenchmarkCPP().run()
