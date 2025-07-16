import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import argparse
import shutil
from pathlib import Path

def setup_voice_directory(voice_name, audio_files_dir):
    """
    Set up a voice directory with the required structure for Tortoise TTS training.
    
    Args:
        voice_name: Name of the voice to create
        audio_files_dir: Directory containing audio files for training
    """
    # Create voice directory
    voice_dir = Path(f"tortoise/voices/{voice_name}")
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files from the input directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(Path(audio_files_dir).glob(f"*{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {audio_files_dir}")
        return False
    
    print(f"Found {len(audio_files)} audio files")
    
    # Copy and rename audio files to voice directory
    for i, audio_file in enumerate(audio_files[:10]):  # Limit to 10 files
        # Convert to wav if needed and copy to voice directory
        output_file = voice_dir / f"{i+1}.wav"
        
        try:
            # Load and resample audio to 22050 Hz (Tortoise requirement)
            audio, sample_rate = torchaudio.load(audio_file)
            
            # Resample if needed
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                audio = resampler(audio)
            
            # Save as wav
            torchaudio.save(output_file, audio, 22050)
            print(f"Processed: {audio_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    print(f"Voice '{voice_name}' setup complete in {voice_dir}")
    return True

def extract_conditioning_latents(voice_name):
    """
    Extract conditioning latents for the trained voice.
    This creates the .pth file needed for inference.
    """
    try:
        tts = TextToSpeech()
        voice_dir = Path(f"tortoise/voices/{voice_name}")
        
        # Load voice samples from the voice directory
        voice_samples = []
        for audio_file in sorted(voice_dir.glob("*.wav")):
            if audio_file.name != f"{voice_name}.pth":  # Skip the latent file if it exists
                audio = load_audio(str(audio_file), 22050)
                voice_samples.append(audio)
        
        if not voice_samples:
            print(f"No voice samples found in {voice_dir}")
            return False
        
        print(f"Loaded {len(voice_samples)} voice samples")
        
        # Get conditioning latents
        voice_samples, conditioning_latents = tts.get_conditioning_latents(voice_samples)
        
        # Save conditioning latents
        latent_file = voice_dir / f"{voice_name}.pth"
        
        torch.save((voice_samples, conditioning_latents), latent_file)
        print(f"Conditioning latents saved to {latent_file}")
        return True
        
    except Exception as e:
        print(f"Error extracting conditioning latents: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train a custom voice for Tortoise TTS")
    parser.add_argument("voice_name", help="Name of the voice to create")
    parser.add_argument("audio_files_dir", help="Directory containing audio files for training")
    parser.add_argument("--extract-latents", action="store_true", 
                       help="Extract conditioning latents after setup")
    
    args = parser.parse_args()
    
    print(f"Setting up voice '{args.voice_name}' from audio files in '{args.audio_files_dir}'")
    
    # Setup voice directory
    if setup_voice_directory(args.voice_name, args.audio_files_dir):
        print("Voice setup completed successfully!")
        
        if args.extract_latents:
            print("Extracting conditioning latents...")
            if extract_conditioning_latents(args.voice_name):
                print("Training complete! You can now use this voice for inference.")
            else:
                print("Failed to extract conditioning latents.")
        else:
            print("Voice setup complete! Run with --extract-latents to finish training.")
    else:
        print("Voice setup failed!")

if __name__ == "__main__":
    main() 