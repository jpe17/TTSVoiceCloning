from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, get_voices
import time
import os
import torch
import torchaudio
import sys

# Check for MPS availability and set device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

tts = TextToSpeech(device=device)

def speak(text, output_file="elon_output.wav", play_audio=True):
    start = time.time()
    
    # Load precomputed voice
    voice_samples, conditioning_latents = load_voice('elonmusk')
    gen = tts.tts_with_preset(
        text=text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset='fast'  # or 'standard', 'high_quality' for better quality
    )
    
    # Save audio using torchaudio
    torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
    print(f"Generated in {time.time() - start:.2f}s")
    print(f"Audio saved to: {output_file}")
    
    # Play audio if requested
    if play_audio:
        if sys.platform == "darwin":  # macOS
            os.system(f"afplay {output_file}")
        elif sys.platform.startswith("linux"):  # Linux
            os.system(f"aplay {output_file}")
        elif sys.platform == "win32":  # Windows
            os.system(f"start {output_file}")
        else:
            print(f"Audio saved to {output_file}. Please play it manually.")

def read_text_from_file(file_path):
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    # Check if text is provided as command line argument
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        # Default text
        text = "Hello, I'm Elon Musk, and you're watching a demo of Tortoise TTS."
    
    # Check if it's a file path
    if text.endswith('.txt'):
        text = read_text_from_file(text)
        if text is None:
            sys.exit(1)
    
    print(f"Generating speech for: '{text}'")
    speak(text)
