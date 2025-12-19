# src/server/tools.py
from langchain_core.tools import tool
from typing import Optional
import json
import time



@tool
def end_call(reason: str = "complete", notes: str = "") -> dict:
    """
    End the call ONLY when user explicitly indicates they're done.

    STRICT REQUIREMENTS:

    1. User must say one of: "bye", "goodbye", "thanks bye", "ok bye"
    2. OR user hangs up
    3. OR no response for extended period
    
    DO NOT end call when user says:
    - "thank you" (just being polite)
    - "ok" or "okay" (just acknowledging)
    - "no" (might have more to say)
    
    IMPORTANT: 
    - ALWAYS say a polite farewell message BEFORE calling this tool
    - Examples: "Thank you for calling! Have a great day!", "Goodbye and take care!"
    - NEVER call this immediately after user says bye - respond first, then call
    
    Args:
        reason: Reason for ending (complete, user_goodbye, no_response)
        notes: Brief summary of the conversation
    """
    # Return a dict that the agent will recognize
    return {
        "action": "END_CALL",
        "reason": reason,
        "notes": notes,
        "timestamp": time.time()
    }



TOOLS = [end_call]





