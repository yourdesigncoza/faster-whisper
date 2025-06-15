#!/usr/bin/env python3
"""
Microphone Listener Script for faster-whisper

This script continuously listens to the microphone, transcribes speech to text
using faster-whisper, and saves the output to a file named 'output.txt'.

Features:
- Real-time microphone recording
- Voice Activity Detection (VAD) to detect speech
- Automatic transcription using faster-whisper
- Continuous output to file
- Configurable recording parameters
- Graceful shutdown with Ctrl+C

Requirements:
- pyaudio (for microphone recording)
- soundfile (for audio file handling)
- faster-whisper (for transcription)
"""

import argparse
import datetime
import io
import os
import signal
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf
from faster_whisper import WhisperModel


class MicrophoneListener:
    """Real-time microphone listener with speech transcription."""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        output_file: str = "output.txt",
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        silence_threshold: float = 0.01,
        min_speech_duration: float = 1.0,
        max_speech_duration: float = 30.0,
        language: Optional[str] = None,
    ):
        """
        Initialize the microphone listener.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run inference on (cpu, cuda)
            compute_type: Compute type (int8, float16, float32)
            output_file: Output file path for transcriptions
            sample_rate: Audio sample rate (16000 recommended for Whisper)
            chunk_duration: Duration of each audio chunk in seconds
            silence_threshold: Threshold for detecting silence (0.0-1.0)
            min_speech_duration: Minimum speech duration to transcribe
            max_speech_duration: Maximum speech duration before forced transcription
            language: Language code (None for auto-detection)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.output_file = Path(output_file)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.language = language
        
        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.speech_detected = False
        self.silence_counter = 0
        self.max_silence_chunks = int(2.0 / chunk_duration)  # 2 seconds of silence
        
        # Whisper model
        self.model = None
        self.transcription_thread = None
        self.transcription_queue = []
        self.queue_lock = threading.Lock()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Stopping microphone listener...")
        self.stop()
        sys.exit(0)
        
    def _load_model(self):
        """Load the Whisper model."""
        print(f"🤖 Loading Whisper model '{self.model_size}' on {self.device}...")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
            
    def _setup_audio_stream(self):
        """Setup the audio input stream."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            print(f"🎤 Audio stream setup: {self.sample_rate}Hz, chunk size: {self.chunk_size}")
        except Exception as e:
            print(f"❌ Failed to setup audio stream: {e}")
            sys.exit(1)
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback function."""
        if self.is_recording:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Calculate RMS (Root Mean Square) for volume detection
            rms = np.sqrt(np.mean(audio_float**2))
            
            # Detect speech based on volume threshold
            if rms > self.silence_threshold:
                if not self.speech_detected:
                    print("🗣️  Speech detected, recording...")
                    self.speech_detected = True
                self.audio_buffer.extend(audio_float)
                self.silence_counter = 0
            else:
                if self.speech_detected:
                    self.silence_counter += 1
                    self.audio_buffer.extend(audio_float)
                    
                    # Check if we should stop recording (silence detected)
                    if self.silence_counter >= self.max_silence_chunks:
                        self._process_audio_buffer()
            
            # Force transcription if buffer is too long
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            if buffer_duration >= self.max_speech_duration:
                print(f"⏰ Max duration reached ({buffer_duration:.1f}s), processing...")
                self._process_audio_buffer()
                
        return (in_data, pyaudio.paContinue)
        
    def _process_audio_buffer(self):
        """Process the current audio buffer for transcription."""
        if not self.audio_buffer:
            return
            
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Only process if we have enough audio
        if buffer_duration >= self.min_speech_duration:
            # Copy buffer and reset
            audio_data = np.array(self.audio_buffer, dtype=np.float32)
            
            # Add to transcription queue
            with self.queue_lock:
                self.transcription_queue.append(audio_data)
                
            print(f"📝 Queued {buffer_duration:.1f}s of audio for transcription")
        
        # Reset buffer and speech detection
        self.audio_buffer = []
        self.speech_detected = False
        self.silence_counter = 0
        
    def _transcription_worker(self):
        """Worker thread for processing transcription queue."""
        while self.is_recording:
            audio_data = None
            
            # Get audio from queue
            with self.queue_lock:
                if self.transcription_queue:
                    audio_data = self.transcription_queue.pop(0)
            
            if audio_data is not None:
                try:
                    # Transcribe audio
                    print("🔄 Transcribing audio...")
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        vad_filter=True,
                        word_timestamps=False
                    )
                    
                    # Collect transcription text
                    transcription = ""
                    for segment in segments:
                        transcription += segment.text
                    
                    if transcription.strip():
                        # Write to output file with timestamp
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        output_line = f"[{timestamp}] {transcription.strip()}\n"
                        
                        # Append to file
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            f.write(output_line)
                        
                        print(f"✅ Transcribed: {transcription.strip()}")
                        print(f"📄 Saved to: {self.output_file}")
                    else:
                        print("🔇 No speech detected in audio")
                        
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
            else:
                # No audio to process, sleep briefly
                time.sleep(0.1)
                
    def start(self):
        """Start the microphone listener."""
        print("🚀 Starting microphone listener...")
        print(f"📁 Output file: {self.output_file.absolute()}")
        print("💡 Speak into your microphone. Press Ctrl+C to stop.")
        print("-" * 50)
        
        # Load model
        self._load_model()
        
        # Setup audio stream
        self._setup_audio_stream()
        
        # Create output file if it doesn't exist
        self.output_file.touch()
        
        # Start recording
        self.is_recording = True
        
        # Start transcription worker thread
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        # Start audio stream
        self.stream.start_stream()
        
        try:
            # Keep main thread alive
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop the microphone listener."""
        print("🛑 Stopping...")
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        # Process any remaining audio in buffer
        if self.audio_buffer:
            self._process_audio_buffer()
            
        # Wait for transcription thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            print("⏳ Finishing transcriptions...")
            self.transcription_thread.join(timeout=10)
            
        print("✅ Stopped successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time microphone transcription")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on")
    parser.add_argument("--compute-type", default="int8",
                       choices=["int8", "float16", "float32"],
                       help="Compute type for inference")
    parser.add_argument("--output", default="output.txt",
                       help="Output file path")
    parser.add_argument("--language", default=None,
                       help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("--silence-threshold", type=float, default=0.01,
                       help="Silence detection threshold (0.0-1.0)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                       help="Minimum speech duration to transcribe (seconds)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                       help="Maximum speech duration before forced transcription")
    
    args = parser.parse_args()
    
    # Create listener
    listener = MicrophoneListener(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        output_file=args.output,
        language=args.language,
        silence_threshold=args.silence_threshold,
        min_speech_duration=args.min_duration,
        max_speech_duration=args.max_duration,
    )
    
    # Start listening
    listener.start()


if __name__ == "__main__":
    main()
