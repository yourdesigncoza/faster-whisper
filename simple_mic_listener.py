#!/usr/bin/env python3
"""
Simple Microphone Listener for faster-whisper

A simplified version that records audio in chunks and transcribes them.
Press Enter to stop recording a chunk and transcribe it.
"""

import datetime
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


class SimpleMicListener:
    """Simple microphone listener with manual control."""
    
    def __init__(self, model_size="base", output_file="output.txt"):
        """Initialize the listener."""
        self.model_size = model_size
        self.output_file = Path(output_file)
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Load Whisper model
        print(f"Loading Whisper model '{model_size}'...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded successfully!")
        
        # Create output file
        self.output_file.touch()
        
    def record_chunk(self, duration=5):
        """Record audio for a specified duration."""
        print(f"🎤 Recording for {duration} seconds...")
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        print("✅ Recording finished!")
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        return audio_float
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio data."""
        print("🔄 Transcribing...")
        
        try:
            segments, info = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect
                vad_filter=True
            )
            
            # Collect transcription
            transcription = ""
            for segment in segments:
                transcription += segment.text
            
            return transcription.strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def save_transcription(self, text):
        """Save transcription to output file."""
        if text:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_line = f"[{timestamp}] {text}\n"
            
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(output_line)
            
            print(f"✅ Transcribed: {text}")
            print(f"📄 Saved to: {self.output_file}")
        else:
            print("🔇 No speech detected")
    
    def run_interactive(self):
        """Run in interactive mode."""
        print("🚀 Simple Microphone Listener")
        print(f"📁 Output file: {self.output_file.absolute()}")
        print("\nCommands:")
        print("  Press Enter to record 5 seconds")
        print("  Type 'q' and press Enter to quit")
        print("  Type a number (e.g., '10') to record for that many seconds")
        print("-" * 50)
        
        try:
            while True:
                user_input = input("\n🎤 Press Enter to record (or 'q' to quit): ").strip()
                
                if user_input.lower() == 'q':
                    break
                
                # Check if user specified duration
                duration = 5  # default
                if user_input.isdigit():
                    duration = int(user_input)
                    duration = max(1, min(60, duration))  # Limit between 1-60 seconds
                
                # Record audio
                audio_data = self.record_chunk(duration)
                
                # Transcribe
                transcription = self.transcribe_audio(audio_data)
                
                # Save to file
                self.save_transcription(transcription)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.cleanup()
    
    def run_continuous(self, chunk_duration=5):
        """Run in continuous mode with fixed chunk duration."""
        print("🚀 Continuous Microphone Listener")
        print(f"📁 Output file: {self.output_file.absolute()}")
        print(f"⏱️  Recording in {chunk_duration}s chunks")
        print("💡 Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Record audio
                audio_data = self.record_chunk(chunk_duration)
                
                # Transcribe
                transcription = self.transcribe_audio(audio_data)
                
                # Save to file
                self.save_transcription(transcription)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.audio.terminate()
        print("✅ Cleanup completed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple microphone transcription")
    parser.add_argument("--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--output", default="output.txt",
                       help="Output file path")
    parser.add_argument("--mode", default="interactive",
                       choices=["interactive", "continuous"],
                       help="Running mode")
    parser.add_argument("--duration", type=int, default=5,
                       help="Chunk duration for continuous mode (seconds)")
    
    args = parser.parse_args()
    
    # Create listener
    listener = SimpleMicListener(
        model_size=args.model,
        output_file=args.output
    )
    
    # Run in selected mode
    if args.mode == "interactive":
        listener.run_interactive()
    else:
        listener.run_continuous(args.duration)


if __name__ == "__main__":
    main()
