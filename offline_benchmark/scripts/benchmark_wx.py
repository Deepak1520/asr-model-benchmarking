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

# --- Configuration ---
MODEL_SIZE = "large-v2"
DEVICE = "cpu" # Force CPU for Mac MPS compatibility
COMPUTE_TYPE = "int8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_FILE = os.path.join(BASE_DIR, "data", "benchmark_results.csv")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "data", "transcriptions", "WhisperX")

os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

# Patch for PyTorch 2.6+ Pickle Security
try:
    import torch.serialization
    import omegaconf
    from omegaconf import listconfig, base, dictconfig, nodes
    import typing
    import collections
    import pyannote.audio.core.model
    import pyannote.audio.core.task
    
    torch.serialization.add_safe_globals([
        listconfig.ListConfig, base.ContainerMetadata, dictconfig.DictConfig,
        base.Node, nodes.AnyNode, base.Metadata, typing.Any, list,
        collections.defaultdict, dict, int, set,
        torch.torch_version.TorchVersion,
        pyannote.audio.core.model.Introspection,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution
    ])
except Exception:
    pass

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
                mem = process.memory_info().rss / (1024 * 1024) 
                self.peak_memory = max(self.peak_memory, mem)
            except:
                pass
            time.sleep(self.interval)

class BenchmarkWX:
    def __init__(self):
        self.results = []
        with open(GROUND_TRUTH_FILE, "r") as f:
            self.ground_truth = json.load(f)

    def get_category(self, path):
        fname = os.path.basename(path).lower()
        if "clean" in fname: return "Clean"
        if "noisy" in fname: return "Noisy"
        if "accented" in fname: return "Accented"
        return "Unknown"

    def run(self):
        print("Benchmarking WhisperX...")
        
        monitor = MemoryMonitor()
        monitor.start()

        start_load = time.time()
        try:
            model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            print(f"Failed to load WhisperX: {e}")
            monitor.stop()
            return
            
        load_time = time.time() - start_load
        print(f"Model Load Time: {load_time:.2f}s")

        files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.wav"), recursive=True))
        files = [f for f in files if "temp" not in f]

        for audio_path in files:
            fname = os.path.basename(audio_path)
            if fname not in self.ground_truth:
                continue

            category = self.get_category(audio_path)
            print(f"Processing {fname} ({category})...")

            # Transcribe
            start_time = time.time()
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=4)
            transcription = " ".join([seg["text"] for seg in result["segments"]]).strip()
            latency = time.time() - start_time

            duration = len(audio) / 16000.0
            rtf = latency / duration if duration > 0 else 0

            # Metrics
            reference = self.ground_truth[fname]
            wer = jiwer.wer(reference, transcription)
            cer = jiwer.cer(reference, transcription)
            
            self._save_transcription(fname, reference, transcription, wer, cer)

            self.results.append({
                "Model": "WhisperX",
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
            time.sleep(5)

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
                if "LoadTime" in existing_df.columns:
                    existing_df = existing_df[existing_df["Model"] != "WhisperX"]
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
    BenchmarkWX().run()
