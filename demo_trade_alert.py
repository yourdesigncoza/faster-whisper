#!/usr/bin/env python3
"""
Demo script showing the exact trade alert format you requested.

This demonstrates the enhanced Telegram trade alert system.
"""

from app.send_telegram import TradeAlert, format_trade_alert

def demo_exact_format():
    """Demo the exact format you specified."""
    print("🎯 Demo: Exact Trade Alert Format")
    print("=" * 50)
    
    # Create the exact alert from your specification
    alert = TradeAlert(
        direction="LONG",
        instrument="gold", 
        condition="if it breaks above 1950",
        confidence=0.8,
        timestamp="2025-06-17 14:08:48"
    )
    
    formatted = format_trade_alert(alert)
    
    print("📱 Telegram Message Format:")
    print(formatted)
    
    print("\n🔍 Raw HTML (what gets sent to Telegram):")
    print(repr(formatted))
    
    print("\n✨ This will appear in Telegram as:")
    print("   • Bold text for Direction, Instrument, Confidence")
    print("   • Italic text for Condition") 
    print("   • Monospace text for Time")
    print("   • Emojis based on confidence level")

if __name__ == "__main__":
    demo_exact_format()
