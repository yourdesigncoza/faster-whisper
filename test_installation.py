#!/usr/bin/env python3
"""
Simple test script to verify faster-whisper installation.
"""

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    try:
        import faster_whisper
        print("✓ faster_whisper imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import faster_whisper: {e}")
        return False
    
    try:
        from faster_whisper import WhisperModel
        print("✓ WhisperModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WhisperModel: {e}")
        return False
    
    try:
        from faster_whisper import BatchedInferencePipeline
        print("✓ BatchedInferencePipeline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import BatchedInferencePipeline: {e}")
        return False
    
    try:
        from faster_whisper import available_models
        print("✓ available_models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import available_models: {e}")
        return False
    
    return True

def test_available_models():
    """Test that we can list available models."""
    print("\nTesting available models...")
    
    try:
        from faster_whisper import available_models
        models = available_models()
        print(f"✓ Found {len(models)} available models")
        print(f"  Sample models: {list(models)[:5]}")
        return True
    except Exception as e:
        print(f"✗ Failed to get available models: {e}")
        return False

def main():
    """Run all tests."""
    print("=== faster-whisper Installation Test ===\n")
    
    tests = [
        test_imports,
        test_available_models,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    exit(main())
