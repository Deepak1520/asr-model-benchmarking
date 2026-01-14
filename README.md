# STT Benchmarking on MacBook Air M3

This project benchmarks three state-of-the-art ASR models on Apple Silicon (M3):
- **Faster-Whisper** (CTranslate2 optimized)
- **WhisperX** (Pytorch/MPS)
- **whisper.cpp** (CoreML/Metal/Arm Neon)

## Prerequisites

- macOS (tested on Sonoma/Sequoia)
- python 3.10+
- `ffmpeg` (required for pydub/audio processing)

## Installation

1.  **Dependencies**
    ```bash
    pip install edge-tts pydub faster-whisper jiwer pandas seaborn matplotlib torch torchaudio
    pip install git+https://github.com/m-bain/whisperx.git
    ```

2.  **Compile whisper.cpp**
    The project expects `whisper.cpp` to be present in the root directory.
    ```bash
    git clone https://github.com/ggerganov/whisper.cpp
    cd whisper.cpp
    
    # Compile with Metal (GPU) support
    make clean
    WHISPER_METAL=1 make -j
    
    # Download Model
    bash ./models/download-ggml-model.sh large-v2
    cd ..
    ```

## Running the Benchmark

1.  **Generate Dataset**
    Generate synthetic dataset (Clean, Noisy, Accented):
    ```bash
    python3 generate_dataset.py
    ```

2.  **Run Benchmark**
   ## Directory Structure
- `scripts/`: Python scripts for benchmarking and visualization.
- `graphs/`: Generated performance charts.
- `dataset/`: Generated audio samples (Clean, Noisy, Accented).
- `whisper.cpp/`: Cloned repository and build files.
- `benchmark_results.csv`: Raw metrics.

## Running the Benchmark

### 1. Generate Dataset
```bash
python3 scripts/generate_dataset.py
```

### 2. Run Benchmarks
Run each model independently:
```bash
# Faster-Whisper
python3 scripts/benchmark_fw.py

# WhisperX
python3 scripts/benchmark_wx.py

# whisper.cpp
python3 scripts/benchmark_cpp.py
```
Note: Ensure you are in the root directory (`try report`).

### 3. Visualize Results
Generate charts in `graphs/`:
```bash
python3 scripts/visualize_results.py
```
.

## CMAKE_ARGS for whisper.cpp (if using CMake)

If you prefer CMake or if directly using `pip install whisper-cpp-python` (not used here, we use binary), you would use:
```bash
CMAKE_ARGS="-DWHISPER_METAL=ON -DWHISPER_CoreML=OFF" pip install whisper-cpp-python
```
For manual build:
```bash
cmake -B build -DWHISPER_METAL=ON
cmake --build build --config Release
```

## Optimization Notes
- **Faster-Whisper**: Uses CTranslate2. On M-series, `int8` quantization on CPU is often fastest and most stable.
- **WhisperX**: Uses PyTorch `mps` backend for acceleration.
- **whisper.cpp**: Uses pure Metal/C++ implementation for high efficiency.
