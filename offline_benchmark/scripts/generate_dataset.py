import asyncio
import base64
import json
import os
import random
import edge_tts
from pydub import AudioSegment
from pydub.generators import WhiteNoise

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "dataset")
GT_FILE = os.path.join(BASE, "data", "ground_truth.json")

# voices
VOICE_US = "en-US-GuyNeural"
VOICE_UK = "en-GB-SoniaNeural"
VOICE_AU = "en-AU-NatashaNeural"
VOICE_IN = "en-IN-NeerjaNeural"

# just some random text to tts
sentences = [
    "The rapid advancement of artificial intelligence has transformed industries.",
    "Climate change represents one of the most significant challenges of our time.",
    "Exploring the vastness of space has always been a dream for humanity.",
    "Sustainable energy sources such as solar and wind power are becoming important.",
    "The history of civilization is filled with stories of triumph and tragedy.",
    "Reading books is a timeless activity that expands our knowledge.",
    "Healthy eating habits combined with regular physical exercise are essential.",
    "The internet has revolutionized communication, connecting people globally.",
    "Art and music have the power to evoke deep emotions.",
    "In the heart of the bustling city, there lies a hidden park.",
    "The concept of time travel has fascinated scientists for generations.",
    "Globalization has interconnected the world in unprecedented ways.",
    "The oceanic depths remain one of the least explored frontiers on Earth.",
    "Artificial intelligence is poised to revolutionize healthcare.",
    "The Great Barrier Reef is a natural wonder that supports marine life.",
    "Architecture has served as a reflection of society's values.",
    "Psychology offers profound insights into the human mind.",
    "Renewable energy technologies have seen dramatic improvements."
]

async def mk_tts(text, voice, out):
    c = edge_tts.Communicate(text, voice)
    await c.save(out)

def to_wav(infile, outfile):
    # 16khz mono for consistency
    x = AudioSegment.from_file(infile)
    x = x.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    x.export(outfile, format="wav")
    return x

def add_noise(sound, snr=15):
    noise = WhiteNoise().to_audio_segment(duration=len(sound))
    # simple gain calculation
    gain = sound.dBFS - snr - noise.dBFS
    return sound.overlay(noise.apply_gain(gain))

async def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    ground_truth = {}
    
    # approx num sentences for length
    files_map = {
        "1min": 6,
        "2min": 12,
        "3min": 18
    }
    
    pool = sentences * 3 # ensure enough text
    
    # Clean
    os.makedirs(os.path.join(OUT_DIR, "clean"), exist_ok=True)
    print("Making clean files...")
    for dur, count in files_map.items():
        text = " ".join(random.sample(pool, count))
        fname = f"clean_{dur}.wav"
        
        tmp = os.path.join(OUT_DIR, "clean", "temp.mp3")
        final = os.path.join(OUT_DIR, "clean", fname)
        
        await mk_tts(text, VOICE_US, tmp)
        to_wav(tmp, final)
        os.remove(tmp)
        
        ground_truth[fname] = text

    # Noisy
    os.makedirs(os.path.join(OUT_DIR, "noisy"), exist_ok=True)
    print("Making noisy files...")
    for dur, count in files_map.items():
        text = " ".join(random.sample(pool, count))
        fname = f"noisy_{dur}.wav"
        
        tmp = os.path.join(OUT_DIR, "noisy", "temp.mp3")
        final = os.path.join(OUT_DIR, "noisy", fname)

        await mk_tts(text, VOICE_US, tmp)
        res = to_wav(tmp, final) # reuse for wave conversion only
        
        # reload to add noise properly
        noisy = add_noise(res, snr=10)
        noisy.export(final, format="wav")
        
        os.remove(tmp)
        ground_truth[fname] = text

    # Accented
    os.makedirs(os.path.join(OUT_DIR, "accented"), exist_ok=True)
    print("Making accented files...")
    
    accents = [
        ("british", VOICE_UK),
        ("australian", VOICE_AU),
        ("indian", VOICE_IN)
    ]
    
    # 0=uk, 1=au, 2=in
    idx = 0
    for dur, count in files_map.items():
        acc_name, voice = accents[idx % 3]
        idx += 1
        
        text = " ".join(random.sample(pool, count))
        fname = f"accented_{dur}_{acc_name}.wav"
        
        tmp = os.path.join(OUT_DIR, "accented", "temp.mp3")
        final = os.path.join(OUT_DIR, "accented", fname)
        
        await mk_tts(text, voice, tmp)
        to_wav(tmp, final)
        os.remove(tmp)
        
        ground_truth[fname] = text

    with open(GT_FILE, "w") as f:
        json.dump(ground_truth, f, indent=2)
        
    print(f"Done. Saved to {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
