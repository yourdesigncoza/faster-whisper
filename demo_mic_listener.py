#!/usr/bin/env python3
"""
Demo script for microphone listener - creates a short test recording.
"""

import datetime
import numpy as np
import pyaudio
from pathlib import Path
from faster_whisper import WhisperModel


def create_demo_recording():
    """Create a short demo recording and transcribe it."""
    print("🎬 Microphone Listener Demo")
    print("=" * 40)
    
    # Setup
    output_file = Path("demo_output.txt")
    sample_rate = 16000
    duration = 3  # seconds
    
    print(f"📁 Output file: {output_file.absolute()}")
    print(f"⏱️  Recording duration: {duration} seconds")
    
    # Initialize audio
    audio = pyaudio.PyAudio()
    
    try:
        # Load model
        print("\n🤖 Loading Whisper model (tiny for demo)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Model loaded!")
        
        # Record audio
        print(f"\n🎤 Recording for {duration} seconds...")
        print("💬 Say something like: 'Hello, this is a test of the microphone listener'")
        print("🔴 Recording starting in 3... 2... 1...")
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        chunk_size = 1024
        for i in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
            
            # Show progress
            progress = (i + 1) / (sample_rate / chunk_size * duration)
            print(f"\r⏱️  Recording: {progress:.0%}", end="", flush=True)
        
        print("\n✅ Recording completed!")
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Check if we got audio
        rms = np.sqrt(np.mean(audio_float**2))
        print(f"📊 Audio RMS level: {rms:.4f}")
        
        if rms < 0.001:
            print("⚠️  Very low audio signal detected")
            print("💡 Try speaking louder or check microphone connection")
        
        # Transcribe
        print("\n🔄 Transcribing audio...")
        segments, info = model.transcribe(
            audio_float,
            language=None,  # Auto-detect
            vad_filter=True
        )
        
        # Collect transcription
        transcription = ""
        for segment in segments:
            transcription += segment.text
        
        transcription = transcription.strip()
        
        # Save to file
        if transcription:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_line = f"[{timestamp}] {transcription}\n"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== Microphone Listener Demo ===\n")
                f.write(f"Demo run at: {timestamp}\n")
                f.write("=" * 40 + "\n\n")
                f.write(output_line)
            
            print(f"✅ Transcription: '{transcription}'")
            print(f"📄 Saved to: {output_file}")
            print(f"🌍 Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        else:
            print("🔇 No speech detected in recording")
            print("💡 Try speaking louder or closer to the microphone")
        
        # Show file contents
        if output_file.exists():
            print(f"\n📖 Contents of {output_file}:")
            print("-" * 40)
            with open(output_file, "r", encoding="utf-8") as f:
                print(f.read())
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        audio.terminate()
    
    print("\n🎉 Demo completed!")
    print("\n📚 Next steps:")
    print("1. Try the interactive listener: python simple_mic_listener.py")
    print("2. Try the advanced listener: python mic_listener.py")
    print("3. Read the guide: MICROPHONE_GUIDE.md")


if __name__ == "__main__":
    create_demo_recording()
