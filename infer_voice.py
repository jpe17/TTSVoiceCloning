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

def speak_text_minimal(text, voice_name="elonmusk", output_file=None, play_audio=True):
    """
    Minimal single-use speech generation optimized for fastest startup
    
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
        print("Initializing minimal TTS for single use...")
        
        # Initialize with absolute minimal settings for fastest startup
        tts = TextToSpeech(
            device=device,
            autoregressive_model_path=None,  # Use default cached model
            diffusion_model_path=None,  # Use default cached model
            vocoder_model_path=None,  # Use default cached model
            enable_redaction=False,  # Skip unnecessary features
            kv_cache=True,  # Enable caching
            use_deepspeed=False,  # Skip DeepSpeed
            half=True if device == 'cuda' else False,  # Use half precision
        )
        
        print(f"Loading voice: {voice_name}")
        voice_samples, conditioning_latents = load_voice(voice_name)
        
        print("Generating speech with minimal settings...")
        
        # Use absolute fastest generation settings
        gen = tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset='ultra_fast',
            k=1,  # Single candidate
            diffusion_iterations=20,  # Minimal iterations
            cond_free=False,  # Skip conditioning-free guidance
            use_deterministic_seed=42,  # Deterministic for speed
        )
        
        # Set output filename if not provided
        if output_file is None:
            output_file = f"{voice_name}_minimal_output.wav"
        
        # Save audio
        torchaudio.save(
            output_file, 
            gen.squeeze(0).cpu(), 
            24000
        )
        generation_time = time.time() - start
        print(f"Total time (including initialization): {generation_time:.2f}s")
        print(f"Audio saved to: {output_file}")
        
        # Play audio if requested
        if play_audio:
            play_audio(output_file)
        
        return output_file
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def speak_text_fastest(text, voice_name="elonmusk", output_file=None, play_audio=True):
    """
    Fastest possible single-use generation with extreme optimizations
    """
    start = time.time()
    
    try:
        print("Initializing fastest TTS mode...")
        
        # Skip model downloads if already cached
        os.environ['TORTOISE_MODELS_DIR'] = os.path.expanduser('~/.cache/tortoise')
        
        # Initialize with fastest possible settings
        tts = TextToSpeech(
            device=device,
            models_dir=None,  # Use cache
            enable_redaction=False,
            kv_cache=True,
            use_deepspeed=False,
            half=True if device == 'cuda' else False,
        )
        
        # Load voice
        voice_samples, conditioning_latents = load_voice(voice_name)
        
        print("Generating with fastest settings...")
        
        # Extreme speed settings
        gen = tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset='ultra_fast',
            k=1,
            diffusion_iterations=15,  # Even fewer iterations
            cond_free=False,
            use_deterministic_seed=42,
        )
        
        if output_file is None:
            output_file = f"{voice_name}_fastest_output.wav"
        
        torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
        
        generation_time = time.time() - start
        print(f"Fastest generation time: {generation_time:.2f}s")
        print(f"Audio saved to: {output_file}")
        
        if play_audio:
            play_audio(output_file)
        
        return output_file
        
    except Exception as e:
        print(f"Error in fastest generation: {e}")
        return None

# Main function for single use
def speak_text(text, voice_name="elonmusk", output_file=None, play_audio=True):
    """
    Main function optimized for single-use generation
    """
    return speak_text_fastest(text, voice_name, output_file, play_audio)

def list_voices():
    """List all available voices"""
    voices = get_voices()
    print("Available voices:")
    for voice in voices:
        print(f"  - {voice}")
    return voices

# Simple usage example optimized for single use
if __name__ == "__main__":
    # Single use - no warm-up needed
    text = "Hello, this is optimized for single use generation."
    speak_text(text, "elonmusk") 