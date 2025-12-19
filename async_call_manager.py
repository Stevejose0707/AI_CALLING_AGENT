# src/api/async_call_manager.py

import asyncio  
import time
import threading
import queue 
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from src.cx_sip.call_manager import CallState as SipCallState
from src.cx_sip.call_manager_pooled import PooledCallManager

class AsyncCallManager:
    """Manages multiple concurrent voice calls with isolated resources"""
    
    def __init__(self, max_concurrent_calls: int = 50, auto_cleanup: bool = True):
        self.active_calls: Dict[str, Dict] = {}
        self.call_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_calls)
        self.max_concurrent_calls = max_concurrent_calls
        self.running = True
        
        if auto_cleanup:
            cleanup_thread = threading.Thread(target=self._auto_cleanup, daemon=True)
            cleanup_thread.start()
        
        print(f"✅ AsyncCallManager initialized with {max_concurrent_calls} max concurrent calls")
    


    async def initiate_call(self, call_id: str, phone_number: str, prompt: str, greeting: str, call_context: Dict = None) -> str:
        """Initiate a new call with optional context"""        
        # Store call info with context
        with self.call_lock:
            if len(self.active_calls) > self.max_concurrent_calls * 3:
                        self.cleanup_old_calls(max_age_seconds=600)  # Aggressive cleanup
                    
            active_count = sum(1 for c in self.active_calls.values() 
                                    if c["status"] in ["pending", "connecting", "in_progress"])
            if active_count >= self.max_concurrent_calls:
                raise Exception(f"Maximum {self.max_concurrent_calls} concurrent calls reached")
        
            self.active_calls[call_id] = {
                "status": "pending",
                "phone_number": phone_number,
                "prompt": prompt,
                "greeting": greeting,
                "start_time": time.time(),
                "call_context": call_context or {},
                "thread": None
            }
        
        # Run call in executor with context
        future = self.executor.submit(
            self._process_call,
            call_id,
            phone_number,
            prompt,
            greeting,
            call_context
        )
        
        with self.call_lock:
            self.active_calls[call_id]["future"] = future
        
        return call_id
    
    def _process_call(self, call_id: str, phone_number: str, prompt: str, greeting: str, call_context: Dict = None) -> Dict:
        """Process a single call in isolation"""
        
        # Define full_context at the beginning to avoid NameError
        full_context = {
            "use_case": "async_api_call",
            "call_id": call_id,
            "api_call_id": call_id
        }
        if call_context:
            full_context.update(call_context)
        
        # Update status
        with self.call_lock:
            if call_id in self.active_calls:
                self.active_calls[call_id]["status"] = "connecting"
        
        call_manager = None
        try:
            # Now full_context is always defined
            call_manager = PooledCallManager(
                custom_prompt=prompt,
                phone_number=phone_number,
                call_context=full_context,
                greeting_message=greeting
            )
            
            # Start SIP
            if not call_manager.start():
                raise Exception("SIP registration failed")
            
            # Update status
            with self.call_lock:
                if call_id in self.active_calls:
                    self.active_calls[call_id]["status"] = "in_progress"
                    self.active_calls[call_id]["call_manager"] = call_manager
            
            # Make the call
            if not call_manager.make_outbound_call():
                raise Exception("Failed to connect call")
            
            # Wait for completion
            while True:
                with call_manager.state_lock:
                    if call_manager.call_state == SipCallState.COMPLETED:
                        break
                time.sleep(0.5)
            
            # Get COMPLETE results
            call_state = getattr(call_manager, 'final_call_state', {})
            conversation = getattr(call_manager, 'final_conversation', [])
            
            # Store EVERYTHING from call_state
            with self.call_lock:
                if call_id in self.active_calls:
                    # Store the complete call_state
                    self.active_calls[call_id].update({
                        "status": "completed",
                        "call_state": call_state,  # Store complete state
                        "conversation": conversation,
                        # Extract key fields for quick access
                        "duration_seconds": call_state.get('duration_seconds', 0),
                        "end_reason": call_state.get('end_reason', 'unknown'),
                        "was_connected": call_state.get('was_connected', False),
                        "total_messages": call_state.get('total_messages', 0),
                        "has_conversation": call_state.get('has_conversation', False),
                        "call_disconnected": call_state.get('call_disconnected', 'unknown'),
                    })
            
            return {
                "success": True,
                "call_state": call_state,
                "conversation": conversation
            }
            
        except Exception as e:
            print(f"Call {call_id} error: {e}")
            
            with self.call_lock:
                if call_id in self.active_calls:
                    self.active_calls[call_id]["status"] = "failed"
                    self.active_calls[call_id]["error"] = str(e)
                    self.active_calls[call_id]["call_state"] = {
                        "call_id": call_id,
                        "was_initiated": False,
                        "phone_number": phone_number,
                        "use_case": full_context.get("use_case", "async_api_call"),
                        "duration_seconds": 0,
                        "end_reason": "error",
                        "error": str(e),
                        "call_status": "failed"
                    }
            
            return {
                "success": False,
                "error": str(e)
            }
            
        finally:
            # Cleanup
            if call_manager:
                call_manager.cleanup()
            
    
    def get_call_status(self, call_id: str) -> Optional[Dict]:
        """Get status of a call"""
        with self.call_lock:
            return self.active_calls.get(call_id)
        
    def _auto_cleanup(self):
        """Background thread to cleanup old calls"""
        while self.running:
            time.sleep(300)  # Every 5 minutes
            removed = self.cleanup_old_calls(max_age_seconds=1800)  # Remove calls older than 30 minutes
            if removed > 0:
                print(f"🧹 Cleaned up {removed} old call records")
    
    def get_all_calls(self) -> Dict:
        """Get all active calls"""
        with self.call_lock:
            return {
                call_id: {
                    "status": info["status"],
                    "phone_number": info["phone_number"],
                    "duration": time.time() - info["start_time"] if info["status"] == "in_progress" else info.get("duration_seconds", 0)
                }
                for call_id, info in self.active_calls.items()
            }
    
    def cancel_call(self, call_id: str) -> bool:
        """Cancel an active call"""
        with self.call_lock:
            if call_id not in self.active_calls:
                return False
            
            call_info = self.active_calls[call_id]
            if call_info["status"] in ["completed", "failed"]:
                return False
            
            # Get call manager and trigger hangup
            call_mgr = call_info.get("call_manager")
            if call_mgr:
                call_mgr.hangup_call()
                call_info["status"] = "cancelled"
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Get call manager statistics"""
        with self.call_lock:
            active = sum(1 for c in self.active_calls.values() if c["status"] == "in_progress")
            completed = sum(1 for c in self.active_calls.values() if c["status"] == "completed")
            failed = sum(1 for c in self.active_calls.values() if c["status"] == "failed")
            
            return {
                "total_calls": len(self.active_calls),
                "active_calls": active,
                "completed_calls": completed,
                "failed_calls": failed,
                # "available_ports": self.websocket_port_pool.qsize(),  <- REMOVE this line
                "available_ports": self.max_concurrent_calls - active,  # <- CHANGE to this
                "max_concurrent": self.max_concurrent_calls
            }
        
    def cleanup_old_calls(self, max_age_seconds: int = 3600):
        """Remove completed/failed calls older than max_age_seconds"""
        with self.call_lock:
            current_time = time.time()
            calls_to_remove = []
            
            for call_id, call_info in self.active_calls.items():
                if call_info["status"] in ["completed", "failed", "cancelled"]:
                    age = current_time - call_info.get("start_time", 0)
                    if age > max_age_seconds:
                        calls_to_remove.append(call_id)
            
            for call_id in calls_to_remove:
                del self.active_calls[call_id]
            
            return len(calls_to_remove)
        
    def shutdown(self):
        """Gracefully shutdown the call manager"""
        print("🛑 Shutting down AsyncCallManager...")
        
        self.running = False  # Stop cleanup thread
        
        # Cancel all active calls
        with self.call_lock:
            for call_id, call_info in self.active_calls.items():
                if call_info["status"] in ["pending", "connecting", "in_progress"]:
                    call_mgr = call_info.get("call_manager")
                    if call_mgr:
                        try:
                            call_mgr.hangup_call()
                        except:
                            pass
        
        # Shutdown executor (remove timeout parameter)
        self.executor.shutdown(wait=True)  # <- REMOVE timeout parameter
        
        # Clear active calls
        with self.call_lock:
            self.active_calls.clear()
        
        print("✅ AsyncCallManager shutdown complete")
#