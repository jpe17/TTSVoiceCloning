import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, get_voices
import time
import sys
import argparse
from pathlib import Path

# Check for MPS availability and set device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

def list_available_voices():
    """List all available voices"""
    voices = get_voices()
    print("Available voices:")
    for voice in voices:
        print(f"  - {voice}")
    return voices

def speak(text, voice_name, output_file=None, play_audio=True, preset='high_quality'):
    """
    Generate speech using a specific voice with high quality settings
    
    Args:
        text: Text to convert to speech
        voice_name: Name of the voice to use
        output_file: Output file path (default: voice_name_output.wav)
        play_audio: Whether to play audio after generation
        preset: Generation preset ('ultra_fast', 'fast', 'standard', 'high_quality')
    """
    start = time.time()
    
    # Initialize TTS
    tts = TextToSpeech(device=device)
    
    try:
        # Load voice
        voice_samples, conditioning_latents = load_voice(voice_name)
        print(f"Loaded voice: {voice_name}")
        
        # Generate speech with high quality
        print(f"Generating high-quality speech with preset: {preset}")
        gen = tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset
        )
        
        # Set output filename if not provided
        if output_file is None:
            output_file = f"{voice_name}_high_quality_output.wav"
        
        # Save audio with high quality settings
        torchaudio.save(
            output_file, 
            gen.squeeze(0).cpu(), 
            24000,
            encoding="PCM_S",
            bits_per_sample=16
        )
        generation_time = time.time() - start
        print(f"Generated high-quality audio in {generation_time:.2f}s")
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

def main():
    parser = argparse.ArgumentParser(description="Generate high-quality speech using Tortoise TTS")
    parser.add_argument("text", nargs='?', help="Text to convert to speech (or file path ending in .txt)")
    parser.add_argument("--voice", "-v", default="elonmusk", help="Voice to use for generation")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--preset", "-p", default="high_quality", 
                       choices=["ultra_fast", "fast", "standard", "high_quality"],
                       help="Generation preset (default: high_quality)")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio after generation")
    parser.add_argument("--list-voices", "-l", action="store_true", help="List available voices")
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        list_available_voices()
        return
    
    # Check if voice exists
    available_voices = get_voices()
    if args.voice not in available_voices:
        print(f"Error: Voice '{args.voice}' not found.")
        print("Available voices:")
        for voice in available_voices:
            print(f"  - {voice}")
        return
    
    # Get text input
    if args.text is None:
        text = input("Enter text to convert to speech: ")
    else:
        text = args.text
    
    # Check if it's a file path
    if text.endswith('.txt'):
        text = read_text_from_file(text)
        if text is None:
            return
    
    print(f"Generating high-quality speech for: '{text}'")
    print(f"Using voice: {args.voice}")
    print(f"Quality preset: {args.preset}")
    
    # Generate speech
    output_file = speak(
        text=text,
        voice_name=args.voice,
        output_file=args.output,
        play_audio=not args.no_play,
        preset=args.preset
    )
    
    if output_file:
        print(f"Success! High-quality audio saved to: {output_file}")

if __name__ == "__main__":
    main() 