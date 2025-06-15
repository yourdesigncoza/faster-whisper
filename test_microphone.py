#!/usr/bin/env python3
"""
Test script to verify microphone setup and audio recording capabilities.
"""

import numpy as np
import pyaudio
import sys
import time


def test_pyaudio_installation():
    """Test if PyAudio is properly installed."""
    print("🔧 Testing PyAudio installation...")
    try:
        audio = pyaudio.PyAudio()
        print("✅ PyAudio installed successfully")
        
        # Get device info
        device_count = audio.get_device_count()
        print(f"📱 Found {device_count} audio devices")
        
        # Find default input device
        try:
            default_input = audio.get_default_input_device_info()
            print(f"🎤 Default input device: {default_input['name']}")
            print(f"   Max input channels: {default_input['maxInputChannels']}")
            print(f"   Default sample rate: {default_input['defaultSampleRate']}")
        except Exception as e:
            print(f"⚠️  Could not get default input device: {e}")
        
        audio.terminate()
        return True
    except Exception as e:
        print(f"❌ PyAudio test failed: {e}")
        return False


def test_microphone_recording():
    """Test basic microphone recording."""
    print("\n🎤 Testing microphone recording...")
    
    try:
        audio = pyaudio.PyAudio()
        
        # Recording parameters
        sample_rate = 16000
        chunk_size = 1024
        duration = 3  # seconds
        
        print(f"📊 Recording parameters:")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Duration: {duration} seconds")
        
        # Open stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print(f"🔴 Recording for {duration} seconds... Speak now!")
        
        frames = []
        for i in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
            
            # Show progress
            progress = (i + 1) / (sample_rate / chunk_size * duration)
            print(f"\r⏱️  Progress: {progress:.1%}", end="", flush=True)
        
        print("\n✅ Recording completed!")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Analyze recorded audio
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Calculate statistics
        rms = np.sqrt(np.mean(audio_float**2))
        max_amplitude = np.max(np.abs(audio_float))
        
        print(f"📈 Audio analysis:")
        print(f"   RMS level: {rms:.4f}")
        print(f"   Max amplitude: {max_amplitude:.4f}")
        print(f"   Audio length: {len(audio_float)} samples ({len(audio_float)/sample_rate:.2f}s)")
        
        if rms > 0.001:
            print("✅ Audio signal detected - microphone is working!")
            return True
        else:
            print("⚠️  Very low audio signal - check microphone connection")
            return False
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False


def test_faster_whisper_import():
    """Test if faster-whisper can be imported."""
    print("\n🤖 Testing faster-whisper import...")
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully")
        
        # Test model loading (tiny model for quick test)
        print("🔄 Testing model loading (tiny model)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ faster-whisper test failed: {e}")
        return False


def list_audio_devices():
    """List all available audio devices."""
    print("\n📱 Available audio devices:")
    try:
        audio = pyaudio.PyAudio()
        
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            device_type = []
            
            if device_info['maxInputChannels'] > 0:
                device_type.append("INPUT")
            if device_info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
            
            print(f"   Device {i}: {device_info['name']}")
            print(f"      Type: {', '.join(device_type)}")
            print(f"      Channels: In={device_info['maxInputChannels']}, Out={device_info['maxOutputChannels']}")
            print(f"      Sample Rate: {device_info['defaultSampleRate']}")
            print()
        
        audio.terminate()
    except Exception as e:
        print(f"❌ Failed to list devices: {e}")


def main():
    """Run all tests."""
    print("🧪 Microphone and Audio Setup Test")
    print("=" * 50)
    
    tests = [
        ("PyAudio Installation", test_pyaudio_installation),
        ("Microphone Recording", test_microphone_recording),
        ("faster-whisper Import", test_faster_whisper_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # List audio devices
    list_audio_devices()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready for microphone transcription.")
        print("\nNext steps:")
        print("1. Run the simple listener: python simple_mic_listener.py")
        print("2. Or run the advanced listener: python mic_listener.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("- Make sure your microphone is connected and working")
        print("- Check audio permissions for your application")
        print("- Try running with sudo if permission issues persist")
    
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit(main())
