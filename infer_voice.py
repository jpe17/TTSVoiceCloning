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

# Global TTS instance to avoid reloading
_tts_instance = None
_voice_cache = {}

def get_tts():
    """Get or create TTS instance (singleton pattern)"""
    global _tts_instance
    if _tts_instance is None:
        print("Initializing TTS model (this happens once)...")
        _tts_instance = TextToSpeech(device=device)
        print("TTS model loaded and ready!")
    return _tts_instance

def get_cached_voice(voice_name):
    """Get cached voice or load it"""
    if voice_name not in _voice_cache:
        print(f"Loading voice: {voice_name}")
        voice_samples, conditioning_latents = load_voice(voice_name)
        _voice_cache[voice_name] = (voice_samples, conditioning_latents)
        print(f"Voice {voice_name} cached for fast access")
    return _voice_cache[voice_name]

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
    Generate speech from text using optimized ultra-fast inference
    
    Args:
        text: Text to convert to speech
        voice_name: Name of the voice to use (default: elonmusk)
        output_file: Output file path (optional)
        play_audio: Whether to play audio after generation (default: True)
    
    Returns:
        str: Path to the generated audio file
    """
    start = time.time()
    
    try:
        # Get cached TTS instance
        tts = get_tts()
        
        # Get cached voice
        voice_samples, conditioning_latents = get_cached_voice(voice_name)
        
        # Generate speech with ultra-fast inference
        print(f"Generating speech...")
        gen = tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset='ultra_fast'
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

def preload_voice(voice_name="elonmusk"):
    """Preload a voice to make subsequent calls faster"""
    print(f"Preloading voice: {voice_name}")
    get_cached_voice(voice_name)
    print(f"Voice {voice_name} ready for instant use!")

# Simple usage example
if __name__ == "__main__":
    # Preload the voice for faster first generation
    preload_voice("elonmusk")
    
    # Example usage - just call the function
    text = "Hello, this is a test of the optimized ultra-fast voice generation system."
    speak_text(text, "elonmusk") 