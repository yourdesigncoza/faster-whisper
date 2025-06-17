import os
import requests
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from .utils.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        # Try direct import when running as script
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from utils.config import config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False

# Configure your bot token and chat ID:
# Option 1: via configuration system (preferred)
if CONFIG_AVAILABLE and config.telegram_token and config.telegram_chat_id:
    TELEGRAM_TOKEN = config.telegram_token
    TELEGRAM_CHAT_ID = config.telegram_chat_id
    TELEGRAM_TIMEOUT = config.telegram_timeout
else:
    # Option 2: via environment variables (fallback)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
    TELEGRAM_TIMEOUT = int(os.getenv("TELEGRAM_TIMEOUT", "10"))

# Basic checks
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
    raise ValueError("Please set the TELEGRAM_TOKEN environment variable or replace the placeholder in the script.")
if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID":
    raise ValueError("Please set the TELEGRAM_CHAT_ID environment variable or replace the placeholder in the script.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeAlert:
    """Represents a trade alert to be sent via Telegram."""
    direction: Optional[str] = None
    instrument: Optional[str] = None
    condition: Optional[str] = None
    confidence: float = 0.0
    timestamp: Optional[str] = None
    raw_text: Optional[str] = None


def is_telegram_configured() -> bool:
    """
    Check if Telegram is properly configured.

    Returns:
        True if Telegram token and chat ID are configured, False otherwise
    """
    return (TELEGRAM_TOKEN and TELEGRAM_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN" and
            TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID")


def test_telegram_connection() -> bool:
    """
    Test the Telegram connection by sending a test message.

    Returns:
        True if connection successful, False otherwise
    """
    if not is_telegram_configured():
        logger.error("❌ Telegram not configured properly")
        return False

    try:
        test_message = "🤖 Telegram connection test - Trade alert system is ready!"
        send_telegram_message(test_message)
        logger.info("✅ Telegram connection test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Telegram connection test failed: {e}")
        return False

def send_telegram_message(message: str, parse_mode: str = "HTML") -> Dict[str, Any]:
    """
    Send a text message to your Telegram account using your bot.

    Args:
        message: The message text to send
        parse_mode: Message formatting mode (HTML, Markdown, or None)

    Returns:
        Response from Telegram API
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(url, data=payload, timeout=TELEGRAM_TIMEOUT)
        if resp.status_code != 200:
            logger.error(f"Failed to send message: {resp.text}")
            raise RuntimeError(f"Failed to send message: {resp.text}")

        logger.info("✅ Message sent successfully to Telegram")
        return resp.json()

    except requests.exceptions.Timeout:
        logger.error(f"Telegram request timed out after {TELEGRAM_TIMEOUT} seconds")
        raise RuntimeError(f"Request timed out after {TELEGRAM_TIMEOUT} seconds")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error sending Telegram message: {e}")
        raise RuntimeError(f"Connection error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending Telegram message: {e}")
        raise RuntimeError(f"Network error: {e}")


def format_trade_alert(alert: TradeAlert) -> str:
    """
    Format a TradeAlert into the desired Telegram message format.

    Args:
        alert: TradeAlert object containing trade information

    Returns:
        Formatted message string
    """
    # Determine emoji based on confidence level
    if alert.confidence >= 0.8:
        emoji = "🚨 ⚡"
        boom_text = "BOOM!"
    elif alert.confidence >= 0.6:
        emoji = "⚡"
        boom_text = "BOOM!"
    else:
        emoji = "📈"
        boom_text = "Boom,"

    # Build the message
    lines = [
        "---",
        f"{emoji} {boom_text} GET READY FOR A TRADE! {emoji.split()[0] if ' ' in emoji else emoji}"
    ]

    # Add trade details
    if alert.direction:
        lines.append(f"Direction: <b>{alert.direction.upper()}</b>")

    if alert.instrument:
        lines.append(f"Instrument: <b>{alert.instrument}</b>")

    if alert.condition:
        lines.append(f"Condition: <i>{alert.condition}</i>")

    lines.append(f"Confidence: <b>{alert.confidence:.1f}</b>")

    # Add timestamp
    timestamp = alert.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Time: <code>{timestamp}</code>")

    lines.append("---")

    return "\n".join(lines)


def send_trade_alert(alert: TradeAlert) -> Dict[str, Any]:
    """
    Send a formatted trade alert to Telegram.

    Args:
        alert: TradeAlert object containing trade information

    Returns:
        Response from Telegram API

    Raises:
        RuntimeError: If Telegram is not configured or message fails to send
    """
    if not is_telegram_configured():
        raise RuntimeError("Telegram is not properly configured. Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.")

    try:
        formatted_message = format_trade_alert(alert)
        logger.info(f"📤 Sending trade alert: {alert.direction} {alert.instrument}")
        return send_telegram_message(formatted_message, parse_mode="HTML")

    except Exception as e:
        logger.error(f"❌ Failed to send trade alert: {e}")
        raise


def send_trade_alert_from_intent(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a trade alert from TradingIntent data structure.

    Args:
        intent_data: Dictionary containing trading intent information

    Returns:
        Response from Telegram API
    """
    # Extract data from intent structure
    if isinstance(intent_data, dict) and "intent_details" in intent_data:
        details = intent_data["intent_details"]
        alert = TradeAlert(
            direction=details.get("direction"),
            instrument=details.get("instrument"),
            condition=details.get("entry_condition"),
            confidence=intent_data.get("confidence", 0.0),
            timestamp=details.get("timestamp"),
            raw_text=details.get("raw_text")
        )
    else:
        # Handle direct TradingIntent object attributes
        alert = TradeAlert(
            direction=getattr(intent_data, "direction", None) or intent_data.get("direction"),
            instrument=getattr(intent_data, "instrument", None) or intent_data.get("instrument"),
            condition=getattr(intent_data, "entry_condition", None) or intent_data.get("entry_condition"),
            confidence=getattr(intent_data, "confidence", 0.0) or intent_data.get("confidence", 0.0),
            timestamp=getattr(intent_data, "timestamp", None) or intent_data.get("timestamp"),
            raw_text=getattr(intent_data, "raw_text", None) or intent_data.get("raw_text")
        )

    return send_trade_alert(alert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a Telegram message or trade alert via bot")
    parser.add_argument("message", nargs="?", help="The message text to send")
    parser.add_argument("--test-alert", action="store_true", help="Send a test trade alert")
    parser.add_argument("--direction", help="Trade direction (long/short)")
    parser.add_argument("--instrument", help="Trading instrument")
    parser.add_argument("--condition", help="Entry condition")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence level (0.0-1.0)")

    args = parser.parse_args()

    try:
        if args.test_alert:
            # Send a test trade alert
            test_alert = TradeAlert(
                direction=args.direction or "LONG",
                instrument=args.instrument or "gold",
                condition=args.condition or "if it breaks above 1950",
                confidence=args.confidence,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            result = send_trade_alert(test_alert)
            print("✅ Test trade alert sent!")
            print(result)

        elif args.message:
            # Send a regular message
            result = send_telegram_message(args.message)
            print("✅ Message sent!")
            print(result)

        else:
            # Send the example trade alert from your specification
            example_alert = TradeAlert(
                direction="LONG",
                instrument="gold",
                condition="if it breaks above 1950",
                confidence=0.8,
                timestamp="2025-06-17 14:08:48"
            )

            result = send_trade_alert(example_alert)
            print("✅ Example trade alert sent!")
            print(result)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"❌ Error: {e}")
        exit(1)
