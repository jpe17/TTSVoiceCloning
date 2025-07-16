import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, get_voices
import time
import sys

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
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {output_file}")
            elif sys.platform.startswith("linux"):  # Linux
                os.system(f"aplay {output_file}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {output_file}")
            else:
                print(f"Audio saved to {output_file}. Please play it manually.")
        
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