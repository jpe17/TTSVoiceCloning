import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, get_voices
import time
import sys
import subprocess

# Better device detection
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_device()
print(f"Using device: {device}")

def play_audio(file_path):
    """Play audio file with multiple fallback methods"""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
        elif sys.platform.startswith("linux"):  # Linux
            # Try multiple audio players
            players = ["aplay", "paplay", "mpg123", "ffplay"]
            for player in players:
                try:
                    subprocess.run([player, file_path], check=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print(f"Audio saved to {file_path}. Please play it manually.")
        elif sys.platform == "win32":  # Windows
            os.system(f"start {file_path}")
        else:
            print(f"Audio saved to {file_path}. Please play it manually.")
    except Exception as e:
        print(f"Could not play audio automatically: {e}")
        print(f"Audio saved to {file_path}. Please play it manually.")

def speak_text(text, voice_name="elonmusk", output_file=None, play_audio=True):
    """
    Generate speech from text using ultra-fast inference with high-quality training data
    
    Args:
        text: Text to convert to speech
        voice_name: Name of the voice to use (default: elonmusk)
        output_file: Output file path (optional)
        play_audio: Whether to play audio after generation (default: True)
    
    Returns:
        str: Path to the generated audio file
    """
    start = time.time()
    
    # Initialize TTS with proper device
    tts = TextToSpeech(device=device)
    
    try:
        # Load voice (this uses the high-quality training data)
        voice_samples, conditioning_latents = load_voice(voice_name)
        print(f"Loaded voice: {voice_name}")
        
        # Generate speech with ultra-fast inference
        print(f"Generating speech with ultra-fast inference...")
        gen = tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset='ultra_fast'  # Almost instant but relies on quality training
        )
        
        # Set output filename if not provided
        if output_file is None:
            output_file = f"{voice_name}_output.wav"
        
        # Save audio
        torchaudio.save(
            output_file, 
            gen.squeeze(0).cpu(), 
            24000
        )
        generation_time = time.time() - start
        print(f"Generated in {generation_time:.2f}s")
        print(f"Audio saved to: {output_file}")
        
        # Play audio if requested
        if play_audio:
            play_audio(output_file)
        
        return output_file
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def list_voices():
    """List all available voices"""
    voices = get_voices()
    print("Available voices:")
    for voice in voices:
        print(f"  - {voice}")
    return voices

# Simple usage example
if __name__ == "__main__":
    # Example usage - just call the function
    text = "Hello, this is a test of the ultra-fast voice generation system."
    speak_text(text, "elonmusk") 