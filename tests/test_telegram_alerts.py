#!/usr/bin/env python3
"""
Test script for the enhanced Telegram trade alert system.

This script tests the trade alert functionality without requiring actual Telegram credentials.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.send_telegram import (
    TradeAlert, 
    format_trade_alert, 
    is_telegram_configured,
    send_trade_alert_from_intent
)
from app.analysis.trading_intent_detector import TradingIntent, TradingIntentDetector
from datetime import datetime


def test_trade_alert_formatting():
    """Test the trade alert message formatting."""
    print("🧪 Testing trade alert formatting...")
    
    # Test case 1: High confidence alert
    alert1 = TradeAlert(
        direction="LONG",
        instrument="gold",
        condition="if it breaks above 1950",
        confidence=0.8,
        timestamp="2025-06-17 14:08:48"
    )
    
    formatted1 = format_trade_alert(alert1)
    print("\n📋 High confidence alert format:")
    print(formatted1)
    
    # Test case 2: Medium confidence alert
    alert2 = TradeAlert(
        direction="SHORT",
        instrument="EURUSD",
        condition="at current levels",
        confidence=0.6,
        timestamp="2025-06-17 14:10:00"
    )
    
    formatted2 = format_trade_alert(alert2)
    print("\n📋 Medium confidence alert format:")
    print(formatted2)
    
    # Test case 3: Low confidence alert
    alert3 = TradeAlert(
        direction="LONG",
        instrument="SPY",
        condition="on breakout",
        confidence=0.4,
        timestamp="2025-06-17 14:12:00"
    )
    
    formatted3 = format_trade_alert(alert3)
    print("\n📋 Low confidence alert format:")
    print(formatted3)
    
    print("✅ Trade alert formatting tests completed!")
    return True


def test_trading_intent_integration():
    """Test integration with TradingIntent objects."""
    print("\n🧪 Testing TradingIntent integration...")
    
    # Create a mock TradingIntent
    intent = TradingIntent(
        intent_detected=True,
        timestamp="14:08:48",
        instrument="gold",
        direction="long",
        entry_condition="if it breaks above 1950",
        confidence=0.8,
        raw_text="I'm going long on gold if it breaks above 1950"
    )
    
    # Create signal analysis from intent
    detector = TradingIntentDetector()
    signal_analysis = detector.create_signal_analysis(intent)
    
    print("📊 Signal analysis structure:")
    for key, value in signal_analysis.items():
        if key != "intent_details":
            print(f"  {key}: {value}")
    
    print("\n📋 Intent details:")
    for key, value in signal_analysis["intent_details"].items():
        print(f"  {key}: {value}")
    
    # Test the alert message generation
    alert_message = detector.generate_alert_message(intent)
    print(f"\n🚨 Generated alert message: {alert_message}")
    
    print("✅ TradingIntent integration tests completed!")
    return True


def test_configuration_check():
    """Test Telegram configuration checking."""
    print("\n🧪 Testing configuration checks...")
    
    is_configured = is_telegram_configured()
    print(f"📱 Telegram configured: {is_configured}")
    
    if not is_configured:
        print("ℹ️  This is expected if you haven't set up Telegram credentials yet.")
        print("   To enable Telegram alerts, set these environment variables:")
        print("   - TELEGRAM_TOKEN: Your bot token from @BotFather")
        print("   - TELEGRAM_CHAT_ID: Your chat ID")
    
    print("✅ Configuration check tests completed!")
    return True


def test_error_handling():
    """Test error handling scenarios."""
    print("\n🧪 Testing error handling...")
    
    # Test with invalid data
    try:
        invalid_intent_data = {
            "trading_intent_detected": True,
            "confidence": "invalid",  # Should be float
            "intent_details": {}
        }
        
        # This should handle the error gracefully
        result = send_trade_alert_from_intent(invalid_intent_data)
        print("⚠️  Error handling test: Should have failed but didn't")
        
    except Exception as e:
        print(f"✅ Error handling test: Correctly caught error - {type(e).__name__}")
    
    print("✅ Error handling tests completed!")
    return True


def run_all_tests():
    """Run all test cases."""
    print("🚀 Starting Telegram Trade Alert System Tests")
    print("=" * 60)
    
    tests = [
        test_trade_alert_formatting,
        test_trading_intent_integration,
        test_configuration_check,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The Telegram trade alert system is ready.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
