import os
import time
import json
import glob
import threading
import psutil
import pandas as pd
import jiwer
from faster_whisper import WhisperModel

# --- Configuration ---
MODEL_SIZE = "large-v2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_FILE = os.path.join(BASE_DIR, "data", "benchmark_results.csv")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "data", "transcriptions", "Faster-Whisper")

os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_memory = 0
        self.thread = threading.Thread(target=self._monitor)

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.peak_memory

    def _monitor(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                mem = process.memory_info().rss / (1024 * 1024) # MB
                self.peak_memory = max(self.peak_memory, mem)
            except:
                pass
            time.sleep(self.interval)

class BenchmarkFW:
    def __init__(self):
        self.results = []
        with open(GROUND_TRUTH_FILE, "r") as f:
            self.ground_truth = json.load(f)

    def get_category(self, path):
        if "long_dataset" in path:
            suffix = "-Long"
        else:
            suffix = ""
            
        if "/clean/" in path: return "Clean" + suffix
        if "/noisy/" in path: return "Noisy" + suffix
        if "/accented/" in path: return "Accented" + suffix
        return "Unknown" + suffix

    def run(self):
        print("Benchmarking Faster-Whisper...")
        
        monitor = MemoryMonitor()
        monitor.start()
        
        start_load = time.time()
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        load_time = time.time() - start_load
        print(f"Model Load Time: {load_time:.2f}s")
        
        files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.wav"), recursive=True))
        files = [f for f in files if "temp" not in f]
        
        print(f"Found {len(files)} files to process.")
        
        for audio_path in files:
            fname = os.path.basename(audio_path)
            if fname not in self.ground_truth:
                continue

            category = self.get_category(audio_path)
            print(f"Processing {fname} ({category})...")
            
            # Transcribe
            start_time = time.time()
            segments, info = model.transcribe(audio_path, beam_size=5)
            transcription = " ".join([s.text for s in segments]).strip()
            latency = time.time() - start_time
            
            duration = info.duration
            rtf = latency / duration if duration > 0 else 0
            
            # Metrics
            reference = self.ground_truth[fname]
            wer = jiwer.wer(reference, transcription)
            cer = jiwer.cer(reference, transcription)
            
            self._save_transcription(fname, reference, transcription, wer, cer)
            
            self.results.append({
                "Model": "Faster-Whisper",
                "Filename": fname,
                "Category": category,
                "Duration": duration,
                "Latency": latency,
                "RTF": rtf,
                "WER": wer,
                "CER": cer,
                "LoadTime": load_time,
                "PeakMemory": 0 
            })
            
            print(f"  -> Latency: {latency:.2f}s, RTF: {rtf:.2f}, WER: {wer:.2f}")
            time.sleep(5) # Cooldown

        peak_memory = monitor.stop()
        print(f"Peak Memory: {peak_memory:.2f} MB")
        
        for r in self.results:
            r["PeakMemory"] = peak_memory

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
                # Cleanup old runs for this model
                if "LoadTime" in existing_df.columns:
                    existing_df = existing_df[existing_df["Model"] != "Faster-Whisper"]
                else:
                    existing_df = pd.DataFrame() # Schema mismatch
                
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            except:
                final_df = new_df
        else:
            final_df = new_df
            
        final_df.to_csv(RESULTS_FILE, index=False)
        print(f"Results updated in {RESULTS_FILE}")

if __name__ == "__main__":
    BenchmarkFW().run()
