#!/usr/bin/env python3
"""
Development tools for faster-whisper project.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def format_code():
    """Format code with black and isort."""
    print("=== Code Formatting ===")
    
    success = True
    success &= run_command("black faster_whisper/ tests/ --check --diff", "Checking code formatting with black")
    success &= run_command("isort faster_whisper/ tests/ --check-only --diff", "Checking import sorting with isort")
    
    if not success:
        print("\n💡 To fix formatting issues, run:")
        print("  black faster_whisper/ tests/")
        print("  isort faster_whisper/ tests/")
    
    return success

def lint_code():
    """Lint code with flake8."""
    print("=== Code Linting ===")
    return run_command("flake8 faster_whisper/ tests/", "Linting code with flake8")

def run_tests():
    """Run tests with pytest."""
    print("=== Running Tests ===")
    return run_command("python -m pytest tests/ -v", "Running tests with pytest")

def run_quick_tests():
    """Run a subset of quick tests."""
    print("=== Running Quick Tests ===")
    return run_command("python -m pytest tests/test_tokenizer.py -v", "Running quick tokenizer tests")

def check_all():
    """Run all checks (format, lint, test)."""
    print("=== Running All Checks ===")
    
    results = []
    results.append(("Formatting", format_code()))
    results.append(("Linting", lint_code()))
    results.append(("Tests", run_tests()))
    
    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        all_passed &= passed
    
    if all_passed:
        print("\n🎉 All checks passed!")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
    
    return all_passed

def fix_formatting():
    """Fix code formatting issues."""
    print("=== Fixing Code Formatting ===")
    
    success = True
    success &= run_command("black faster_whisper/ tests/", "Formatting code with black")
    success &= run_command("isort faster_whisper/ tests/", "Sorting imports with isort")
    
    return success

def test_microphone():
    """Test microphone setup."""
    print("=== Testing Microphone Setup ===")
    return run_command("python test_microphone.py", "Testing microphone and audio setup")

def demo_microphone():
    """Run microphone demo."""
    print("=== Running Microphone Demo ===")
    return run_command("timeout 30 python demo_mic_listener.py", "Running microphone demo")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Development tools for faster-whisper")
    parser.add_argument("command", choices=[
        "format", "lint", "test", "quick-test", "check-all", "fix-format", "test-mic", "demo-mic"
    ], help="Command to run")

    args = parser.parse_args()
    
    # Ensure we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Please activate the virtual environment first: source venv/bin/activate")
        return 1
    
    commands = {
        "format": format_code,
        "lint": lint_code,
        "test": run_tests,
        "quick-test": run_quick_tests,
        "check-all": check_all,
        "fix-format": fix_formatting,
        "test-mic": test_microphone,
        "demo-mic": demo_microphone,
    }
    
    success = commands[args.command]()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
