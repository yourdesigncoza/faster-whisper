"""
Trading Intent Detector using LLM-based analysis.

This module implements an intent-focused approach to detect when a trader
expresses clear intent to enter a trade, replacing the keyword-based system
that generates too many false positives.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .openai_client import OpenAIAnalyzer


@dataclass
class TradingIntent:
    """Represents detected trading intent."""
    intent_detected: bool
    timestamp: Optional[str] = None
    instrument: Optional[str] = None
    direction: Optional[str] = None  # "long" or "short"
    entry_condition: Optional[str] = None
    confidence: float = 0.0
    raw_text: Optional[str] = None


class TradingIntentDetector:
    """
    Detects explicit trading intent using structured LLM prompts.
    
    This detector looks for clear expressions of "I'm going to trade" rather than
    general trading discussion, significantly reducing false positives.
    """
    
    SYSTEM_PROMPT = """You are a real-time trading assistant. Your job is to read incoming transcript text from a live day-trading stream and determine if, in that snippet, the trader expresses clear intent to enter a trade.

Look for EXPLICIT language indicating the trader is about to take action, such as:
- "I'm going to buy/sell..."
- "I'll go long/short if..."
- "Taking a position at..."
- "Entering here..."
- "I'm buying/selling now..."
- "Going long/short on this..."

DO NOT trigger on general trading discussion like:
- Market analysis ("this looks bullish")
- Educational content ("when you see this pattern...")
- Hypothetical scenarios ("you could buy here")
- Past trades ("I bought yesterday")
- General observations ("nice setup")

If there is no clear intent to enter a trade RIGHT NOW or under specific conditions, reply with:
{"intent_detected": false}

When clear intent is detected, reply in JSON format with:
{
  "intent_detected": true,
  "timestamp": "HH:MM:SS from transcript if available",
  "instrument": "trading instrument if mentioned",
  "direction": "long or short if detectable",
  "entry_condition": "specific condition mentioned, e.g., 'if it breaks above X', 'at this level', etc.",
  "confidence": 0.8
}

Be conservative - only flag genuine intent to trade, not general discussion."""

    def __init__(self, openai_analyzer: Optional[OpenAIAnalyzer] = None):
        """
        Initialize the trading intent detector.
        
        Args:
            openai_analyzer: OpenAI analyzer instance. If None, creates a new one.
        """
        self.openai_analyzer = openai_analyzer or OpenAIAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def detect_intent(self, transcript_text: str, context: Optional[str] = None) -> TradingIntent:
        """
        Detect trading intent in transcript text.
        
        Args:
            transcript_text: The transcript text to analyze
            context: Optional context from previous analyses
            
        Returns:
            TradingIntent object with detection results
        """
        try:
            # Prepare the user prompt
            user_prompt = self._build_user_prompt(transcript_text, context)
            
            # Get LLM response
            response = self._get_llm_response(user_prompt)
            
            # Parse the response
            intent = self._parse_response(response, transcript_text)
            
            self.logger.debug(f"Intent detection result: {intent.intent_detected}, confidence: {intent.confidence}")
            
            return intent
            
        except Exception as e:
            self.logger.error(f"Error in intent detection: {e}")
            return TradingIntent(intent_detected=False)
    
    def _build_user_prompt(self, transcript_text: str, context: Optional[str] = None) -> str:
        """Build the user prompt for intent detection."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Previous context: {context}")
            prompt_parts.append("")
        
        prompt_parts.append("Here is the latest transcript segment. Analyze it and respond according to the instructions:")
        prompt_parts.append("")
        prompt_parts.append(transcript_text)
        
        return "\n".join(prompt_parts)
    
    def _get_llm_response(self, user_prompt: str) -> str:
        """Get response from the LLM."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.openai_analyzer._make_request(
            messages=messages,
            max_tokens=300,
            temperature=0.1  # Low temperature for consistent responses
        )
    
    def _parse_response(self, response: str, original_text: str) -> TradingIntent:
        """Parse the LLM response into a TradingIntent object."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                # If no JSON found, assume no intent
                return TradingIntent(intent_detected=False, raw_text=original_text)
            
            data = json.loads(json_match.group())
            
            return TradingIntent(
                intent_detected=data.get("intent_detected", False),
                timestamp=data.get("timestamp"),
                instrument=data.get("instrument"),
                direction=data.get("direction"),
                entry_condition=data.get("entry_condition"),
                confidence=data.get("confidence", 0.0),
                raw_text=original_text
            )
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}. Response: {response}")
            return TradingIntent(intent_detected=False, raw_text=original_text)
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return TradingIntent(intent_detected=False, raw_text=original_text)
    
    def generate_alert_message(self, intent: TradingIntent) -> str:
        """
        Generate an alert message based on detected intent.
        
        Args:
            intent: The detected trading intent
            
        Returns:
            Formatted alert message
        """
        if not intent.intent_detected:
            return ""
        
        # Base message with confidence level
        if intent.confidence >= 0.8:
            base_msg = "🚨 BOOM! GET READY FOR A TRADE! 🚨"
        elif intent.confidence >= 0.6:
            base_msg = "⚡ BOOM! GET READY FOR A TRADE! ⚡"
        else:
            base_msg = "📈 Boom, Get Ready for a trade!"
        
        # Add details if available
        details = []
        if intent.direction:
            details.append(f"Direction: {intent.direction.upper()}")
        if intent.instrument:
            details.append(f"Instrument: {intent.instrument}")
        if intent.entry_condition:
            details.append(f"Condition: {intent.entry_condition}")
        
        if details:
            return f"{base_msg} {' | '.join(details)}"
        else:
            return f"{base_msg} Trading intent detected!"
    
    def create_signal_analysis(self, intent: TradingIntent) -> Dict[str, Any]:
        """
        Create a signal analysis dictionary compatible with the existing system.
        
        Args:
            intent: The detected trading intent
            
        Returns:
            Signal analysis dictionary
        """
        signal_strength = 0
        if intent.intent_detected:
            # Convert confidence to 0-10 scale
            signal_strength = min(10, max(3, int(intent.confidence * 10)))
        
        return {
            "trading_intent_detected": intent.intent_detected,
            "signal_strength": signal_strength,
            "confidence": intent.confidence,
            "detected_signals": [
                f"Intent: {intent.direction or 'unspecified'} position",
                f"Instrument: {intent.instrument or 'unspecified'}",
                f"Condition: {intent.entry_condition or 'immediate'}"
            ] if intent.intent_detected else [],
            "alert_message": self.generate_alert_message(intent) if intent.intent_detected else "",
            "analysis_timestamp": datetime.now().isoformat(),
            "intent_details": {
                "timestamp": intent.timestamp,
                "instrument": intent.instrument,
                "direction": intent.direction,
                "entry_condition": intent.entry_condition,
                "raw_text": intent.raw_text
            }
        }
