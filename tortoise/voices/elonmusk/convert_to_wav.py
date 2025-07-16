import os
import torch
import torchaudio
from pathlib import Path
import glob

def convert_to_wav():
    """Convert all audio files in current directory to numbered WAV files"""
    
    # Get current directory
    current_dir = Path(".")
    
    # Audio file extensions to convert
    audio_extensions = ['*.mp3', '*.m4a', '*.flac', '*.aac', '*.ogg', '*.wma', '*.aiff']
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(ext))
        audio_files.extend(glob.glob(ext.upper()))
    
    print(f"Found {len(audio_files)} audio files to convert")
    
    # Convert each file
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"Converting {audio_file} to {i}.wav...")
            
            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample to 22050 Hz (Tortoise requirement)
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                audio = resampler(audio)
            
            # Save as numbered WAV file
            output_file = f"{i}.wav"
            torchaudio.save(output_file, audio, 22050)
            
            print(f"✓ Converted to {output_file}")
            
        except Exception as e:
            print(f"✗ Error converting {audio_file}: {e}")
            continue
    
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_wav() 