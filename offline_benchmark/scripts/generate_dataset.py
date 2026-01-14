import asyncio
import json
import os
import random
import edge_tts
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "dataset")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")

VOICES = {
    "neutral": "en-US-GuyNeural",
    "uk": "en-GB-SoniaNeural",
    "au": "en-AU-NatashaNeural",
    "in": "en-IN-NeerjaNeural"
}

# Pool of sentences to combine
SENTENCES = [
    "The rapid advancement of artificial intelligence has transformed industries, allowing for automation and efficiency that was previously unimaginable in the modern world of technology and innovation.",
    "Climate change represents one of the most significant challenges of our time, requiring global cooperation and immediate action to mitigate its devastating effects on our planet's ecosystem.",
    "Exploring the vastness of space has always been a dream for humanity, driving us to develop new technologies and push the boundaries of what is scientifically possible.",
    "Sustainable energy sources such as solar and wind power are becoming increasingly important as we strive to reduce our carbon footprint and preserve the environment for future generations.",
    "The history of civilization is filled with stories of triumph and tragedy, teaching us valuable lessons about resilience, culture, and the enduring spirit of the human race.",
    "Reading books is a timeless activity that expands our knowledge, improves vocabulary, and transports us to different worlds, fostering creativity and empathy in readers of all ages.",
    "Healthy eating habits combined with regular physical exercise are essential for maintaining a strong immune system and preventing chronic diseases that can affect our quality of life.",
    "The internet has revolutionized communication, connecting people from all corners of the globe and making information accessible to anyone with a computer or a smartphone.",
    "Art and music have the power to evoke deep emotions, transcend cultural barriers, and bring people together in a shared experience of beauty and artistic expression.",
    "In the heart of the bustling city, amidst the cacophony of traffic and the endless sea of people, there lies a hidden park that offers a sanctuary of peace and tranquility.",
    "The concept of time travel has fascinated scientists and science fiction writers for generations, sparking countless debates about the theoretical possibilities and paradoxes that such a journey would entail.",
    "Globalization has interconnected the world in unprecedented ways, facilitating the exchange of goods, services, and ideas across international borders.",
    "The oceanic depths remain one of the least explored frontiers on Earth, housing a diverse array of marine life that thrives in extreme conditions of pressure and darkness.",
    "Artificial intelligence is poised to revolutionize healthcare by enabling more accurate diagnoses, personalized treatment plans, and the accelerated discovery of new drugs.",
    "The Great Barrier Reef, the world's largest coral reef system, is a natural wonder that supports a staggering variety of marine life.",
    "Throughout history, architecture has served as a reflection of society's values, technological advancements, and artistic aspirations.",
    "The study of psychology offers profound insights into the human mind, exploring the complex interplay between biological factors, environmental influences, and personal experiences.",
    "Renewable energy technologies, such as solar panels and wind turbines, have seen dramatic improvements in efficiency and cost-effectiveness in recent years."
]

async def generate_tts(text, voice, outfile):
    comm = edge_tts.Communicate(text, voice)
    await comm.save(outfile)

def convert_to_wav(infile, outfile):
    """Normalize to 16kHz mono 16-bit WAV."""
    audio = AudioSegment.from_file(infile)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(outfile, format="wav")
    return audio

def add_white_noise(audio_segment, snr_db=15):
    """Overlay white noise."""
    noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))
    target_noise_db = audio_segment.dBFS - snr_db
    noise = noise.apply_gain(target_noise_db - noise.dBFS)
    return audio_segment.overlay(noise)

async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ground_truth = {}
    
    # Define target durations (approximate word counts / sentence counts)
    # Using simple sentence concatenation logic: 2 sentences ~ 20s. 
    # 1min (60s) -> 6 sentences
    # 2min (120s) -> 12 sentences
    # 3min (180s) -> 18 sentences
    DURATIONS = {
        "1min": 6,
        "2min": 12,
        "3min": 18
    }
    
    # Create subdirectories
    subdirs = ["clean", "noisy", "accented"]
    for d in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)
    
    # Ensure we have enough sentences for sampling 18
    # Duplicate the list to avoid running out of population
    SENTENCES_POOL = SENTENCES * 2

    # 1. Clean Samples
    print("Generating Clean samples...")
    for label, count in DURATIONS.items():
        subset = random.sample(SENTENCES_POOL, count)
        text = " ".join(subset)
        
        filename = f"clean_{label}.wav"
        temp_file = os.path.join(OUTPUT_DIR, "clean", f"temp_clean_{label}.mp3")
        final_file = os.path.join(OUTPUT_DIR, "clean", filename)

        await generate_tts(text, VOICES["neutral"], temp_file)
        convert_to_wav(temp_file, final_file)
        os.remove(temp_file)
        
        ground_truth[filename] = text

    # 2. Noisy Samples
    print("Generating Noisy samples...")
    for label, count in DURATIONS.items():
        subset = random.sample(SENTENCES_POOL, count)
        text = " ".join(subset)
        
        filename = f"noisy_{label}.wav"
        temp_file = os.path.join(OUTPUT_DIR, "noisy", f"temp_noisy_{label}.mp3")
        final_file = os.path.join(OUTPUT_DIR, "noisy", filename)
        
        await generate_tts(text, VOICES["neutral"], temp_file)
        
        audio = AudioSegment.from_file(temp_file)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        noisy = add_white_noise(audio, snr_db=10)
        noisy.export(final_file, format="wav")
        
        os.remove(temp_file)
        ground_truth[filename] = text

    # 3. Accented Samples (Rotate accents)
    print("Generating Accented samples...")
    accent_list = [("british", VOICES["uk"]), ("australian", VOICES["au"]), ("indian", VOICES["in"])]
    
    # Map durations to accents
    accent_map = {
        "1min": 0,
        "2min": 1,
        "3min": 2
    }
    
    for label, count in DURATIONS.items():
        subset = random.sample(SENTENCES_POOL, count)
        text = " ".join(subset)
        
        idx = accent_map[label]
        acc_name, voice = accent_list[idx]
        
        filename = f"accented_{label}_{acc_name}.wav"
        temp_file = os.path.join(OUTPUT_DIR, "accented", f"temp_acc_{label}.mp3")
        final_file = os.path.join(OUTPUT_DIR, "accented", filename)
        
        await generate_tts(text, voice, temp_file)
        convert_to_wav(temp_file, final_file)
        os.remove(temp_file)
        
        ground_truth[filename] = text

    # Save Ground Truth
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Dataset generated in '{OUTPUT_DIR}'")
    print(f"Ground truth saved to '{GROUND_TRUTH_FILE}'")

if __name__ == "__main__":
    asyncio.run(main())
