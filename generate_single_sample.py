import asyncio
import edge_tts
import random
from pydub import AudioSegment
import os

# Configuration
TASKS = [
    {"name": "sample_2min", "count": 12},
    {"name": "sample_3min", "count": 18}
]

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

# Duplicate sentences to ensure enough unique sampling for longer durations
SENTENCES = SENTENCES * 3

async def main():
    for task in TASKS:
        output_wav = f"{task['name']}.wav"
        output_txt = f"{task['name']}.txt"
        
        # Select sentences
        subset = random.sample(SENTENCES, task['count'])
        text = " ".join(subset)
        
        print(f"Generating {task['name']} ({len(text.split())} words)...")
        
        # Generate TTS
        voice = "en-US-GuyNeural"
        comm = edge_tts.Communicate(text, voice)
        temp_mp3 = f"temp_{task['name']}.mp3"
        await comm.save(temp_mp3)
        
        # Convert to WAV
        audio = AudioSegment.from_file(temp_mp3)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_wav, format="wav")
        
        # Clean up
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            
        # Save Ground Truth
        with open(output_txt, "w") as f:
            f.write(text)
            
        print(f"✅ Generated '{output_wav}'")
        print(f"✅ Saved ground truth to '{output_txt}'")

if __name__ == "__main__":
    asyncio.run(main())
