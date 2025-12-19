# src/api/models.py

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class CallRequest(BaseModel):
    phone_number: str
    prompt: str
    greeting_message: Optional[str] = "Hello, how can I help you today?"
    call_context: Optional[Dict[str, Any]] = {}  # Add this for custom context

class CallInitResponse(BaseModel):
    call_id: str
    status: str
    message: str
    phone_number: str

class CallStatusResponse(BaseModel):
    call_id: str
    status: str  # pending, connecting, in_progress, completed, failed
    phone_number: str
    duration_seconds: Optional[float] = None
    conversation: Optional[List] = None
    end_reason: Optional[str] = None
    error: Optional[str] = None
    # Add complete metadata
    call_metadata: Optional[Dict[str, Any]] = None


class CallCompleteMetadata(BaseModel):
    """Complete call metadata matching call_state from CallManager"""
    call_id: str
    was_initiated: bool
    phone_number: str
    use_case: str
    duration_seconds: float
    end_reason: str
    call_disconnected: str  # user_hangup or agent_hangup
    greeting_message: Optional[str]
    call_start_time: Optional[float]
    call_end_time: Optional[float]
    was_connected: bool
    call_initiated_at: Optional[str]
    connected_at: Optional[str]
    total_messages: int
    has_conversation: bool
    noise_reduction: str
    vad_type: str
    call_context: Dict[str, Any]
    call_status: str

class CallResponse(BaseModel):
    """Complete response with all metadata"""
    success: bool
    # Core info
    call_id: str
    phone_number: str
    status: str
    
    # Complete metadata
    was_initiated: bool
    use_case: str
    duration_seconds: float
    end_reason: str
    call_disconnected: str
    greeting_message: Optional[str]
    call_start_time: Optional[float]
    call_end_time: Optional[float]
    was_connected: bool
    call_initiated_at: Optional[str]
    connected_at: Optional[str]
    total_messages: int
    has_conversation: bool
    noise_reduction: str
    vad_type: str
    call_context: Dict[str, Any]
    
    # Conversation
    conversation: List
    
    # Error if any
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    active_calls: int
    total_calls: int