import os
import torch
import torchaudio
import torchaudio.transforms as T
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import argparse
import shutil
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """High-quality audio processing for voice training"""
    
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        
        # High quality settings
        self.min_duration = 3.0  # Minimum 3 seconds
        self.max_duration = 30.0  # Maximum 30 seconds
        self.min_amplitude = 0.01  # Minimum amplitude threshold
        self.noise_reduction = True
        self.normalize_audio = True
    
    def load_and_preprocess(self, audio_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess audio with high quality settings"""
        try:
            # Load audio
            audio, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample with high quality
            if sample_rate != self.target_sr:
                resampler = T.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sr,
                    dtype=audio.dtype
                )
                audio = resampler(audio)
            
            # Quality checks
            if not self._check_audio_quality(audio):
                logger.warning(f"Audio quality check failed for {audio_path.name}")
                return None
            
            # Apply preprocessing
            audio = self._preprocess_audio(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def _check_audio_quality(self, audio: torch.Tensor) -> bool:
        """Check if audio meets quality standards"""
        duration = audio.shape[1] / self.target_sr
        
        # Check duration
        if duration < self.min_duration or duration > self.max_duration:
            logger.info(f"Audio duration {duration:.2f}s outside acceptable range")
            return False
        
        # Check amplitude
        max_amplitude = torch.max(torch.abs(audio))
        if max_amplitude < self.min_amplitude:
            logger.info(f"Audio amplitude too low: {max_amplitude:.4f}")
            return False
        
        # Check for silence
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms < 0.001:
            logger.info("Audio appears to be mostly silence")
            return False
        
        return True
    
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply high-quality preprocessing"""
        # Normalize audio
        if self.normalize_audio:
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Apply noise reduction
        if self.noise_reduction:
            # Simple noise gate
            noise_threshold = 0.01
            audio = torch.where(torch.abs(audio) < noise_threshold, 
                              torch.zeros_like(audio), audio)
        
        # Apply gentle high-pass filter to remove low-frequency noise
        # Using FFT-based filtering instead of HighpassBiquad
        if audio.shape[1] > 1000:  # Only if audio is long enough
            try:
                # Simple high-pass filter using FFT
                fft = torch.fft.rfft(audio)
                freqs = torch.fft.rfftfreq(audio.shape[1], 1/self.target_sr)
                
                # Create high-pass filter (cutoff at 80 Hz)
                cutoff_freq = 80.0
                filter_response = torch.ones_like(freqs)
                filter_response[freqs < cutoff_freq] = 0.1  # Attenuate low frequencies
                
                # Apply filter
                fft_filtered = fft * filter_response
                audio = torch.fft.irfft(fft_filtered, n=audio.shape[1])
                
            except Exception as e:
                logger.warning(f"High-pass filtering failed: {e}, continuing without filter")
        
        return audio

def setup_voice_directory(voice_name: str, audio_files_dir: str) -> bool:
    """
    Set up a voice directory with high-quality processing for Tortoise TTS training.
    
    Args:
        voice_name: Name of the voice to create
        audio_files_dir: Directory containing audio files for training
    """
    # Create voice directory
    voice_dir = Path(f"tortoise/voices/{voice_name}")
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize audio processor
    processor = AudioProcessor()
    
    # Get all audio files from the input directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(Path(audio_files_dir).glob(f"*{ext}")))
        audio_files.extend(list(Path(audio_files_dir).glob(f"*{ext.upper()}")))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_files_dir}")
        return False
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process audio files with quality checks
    processed_count = 0
    max_files = 25  # Process up to 25 files for high quality
    
    for i, audio_file in enumerate(audio_files[:max_files]):
        logger.info(f"Processing {i+1}/{min(len(audio_files), max_files)}: {audio_file.name}")
        
        # Process audio with high quality
        processed_audio = processor.load_and_preprocess(audio_file)
        
        if processed_audio is not None:
            output_file = voice_dir / f"{processed_count+1}.wav"
            
            try:
                # Save with high quality settings
                torchaudio.save(
                    output_file, 
                    processed_audio, 
                    processor.target_sr,
                    encoding="PCM_S",
                    bits_per_sample=16
                )
                processed_count += 1
                logger.info(f"Successfully processed: {audio_file.name} -> {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error saving {output_file}: {e}")
                continue
        else:
            logger.warning(f"Skipped {audio_file.name} due to quality issues")
    
    if processed_count == 0:
        logger.error("No audio files were successfully processed!")
        return False
    
    logger.info(f"Successfully processed {processed_count} audio files")
    logger.info(f"Voice '{voice_name}' setup complete in {voice_dir}")
    return True

def extract_conditioning_latents(voice_name: str) -> bool:
    """
    Extract conditioning latents for the trained voice with high quality.
    This creates the .pth file needed for inference.
    """
    try:
        logger.info("Initializing TextToSpeech model...")
        tts = TextToSpeech()
        voice_dir = Path(f"tortoise/voices/{voice_name}")
        
        # Load voice samples from the voice directory
        voice_samples = []
        audio_files = sorted(voice_dir.glob("*.wav"))
        
        if not audio_files:
            logger.error(f"No voice samples found in {voice_dir}")
            return False
        
        logger.info(f"Loading {len(audio_files)} voice samples...")
        
        for audio_file in audio_files:
            if audio_file.name != f"{voice_name}.pth":  # Skip the latent file if it exists
                try:
                    audio = load_audio(str(audio_file), 22050)
                    voice_samples.append(audio)
                    logger.debug(f"Loaded: {audio_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {audio_file.name}: {e}")
                    continue
        
        if not voice_samples:
            logger.error("No voice samples could be loaded!")
            return False
        
        logger.info(f"Successfully loaded {len(voice_samples)} voice samples")
        
        # Get conditioning latents with error handling
        logger.info("Extracting conditioning latents...")
        try:
            voice_samples, conditioning_latents = tts.get_conditioning_latents(voice_samples)
        except Exception as e:
            logger.error(f"Error during latent extraction: {e}")
            return False
        
        # Save conditioning latents
        latent_file = voice_dir / f"{voice_name}.pth"
        
        try:
            torch.save((voice_samples, conditioning_latents), latent_file)
            logger.info(f"Conditioning latents saved to {latent_file}")
            
            # Verify the saved file
            if latent_file.exists() and latent_file.stat().st_size > 0:
                logger.info("Latent file verification successful")
                return True
            else:
                logger.error("Latent file verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error saving latent file: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error extracting conditioning latents: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train a custom voice for Tortoise TTS with high quality")
    parser.add_argument("voice_name", help="Name of the voice to create")
    parser.add_argument("audio_files_dir", help="Directory containing audio files for training")
    
    args = parser.parse_args()
    
    logger.info(f"Setting up voice '{args.voice_name}' from audio files in '{args.audio_files_dir}'")
    logger.info("Using high-quality processing mode")
    
    # Setup voice directory
    if setup_voice_directory(args.voice_name, args.audio_files_dir):
        logger.info("Voice setup completed successfully!")
        
        logger.info("Extracting conditioning latents...")
        if extract_conditioning_latents(args.voice_name):
            logger.info("Training complete! You can now use this voice for inference.")
            logger.info(f"Voice files are located in: tortoise/voices/{args.voice_name}/")
        else:
            logger.error("Failed to extract conditioning latents.")
    else:
        logger.error("Voice setup failed!")

if __name__ == "__main__":
    main() 