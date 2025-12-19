# src/cx_sip/call_manager_pooled.py

import threading
from .call_manager import CallManager
from .sip_connection_pool import get_sip_pool


class PooledCallManager(CallManager):
    """CallManager that uses SIP connection pool"""
    
    def __init__(self, *args, **kwargs):
        # Remove websocket_url from kwargs if present
        kwargs.pop('websocket_url', None)
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Get the global SIP pool
        self.sip_pool = get_sip_pool()
        self.pooled_connection = None
        self.is_using_pool = False
    
    def start(self):
        """Start using pooled connection instead of creating new one"""
        print("🔄 Getting SIP connection from pool...")
        
        # Get connection from pool instead of creating new
        self.pooled_connection = self.sip_pool.get_connection(timeout=10)
        
        if not self.pooled_connection:
            print("❌ No available SIP connections in pool")
            # Fallback to creating new connection
            return super().start()
        
        print("✅ Got pooled SIP connection")
        
        # Use the pooled connection
        self.sip_bridge = self.pooled_connection
        self.is_using_pool = True
        
        # Start listening thread for this connection
        listen_thread = threading.Thread(
            target=self.sip_bridge.listen_for_messages
        )
        listen_thread.daemon = True
        listen_thread.start()
        
        # No need for refresh thread - pool handles it
        
        return True
    
    def cleanup(self):
        """Return connection to pool instead of destroying it"""
        print("\n🧹 Cleaning up pooled call manager...")
        
        # Set flags
        self.running = False
        self.hangup_event.set()
        
        # Clean up RTP handler
        if self.rtp_handler:
            self.rtp_handler.close()
            self.rtp_handler = None
        
        # Return SIP connection to pool if we're using pool
        if self.is_using_pool and self.pooled_connection:
            print("♻️ Returning SIP connection to pool")
            self.sip_pool.return_connection(self.pooled_connection)
            self.pooled_connection = None
            self.sip_bridge = None
        else:
            # If not using pool, do normal cleanup
            if self.sip_bridge:
                self.sip_bridge.cleanup()
        
        print("✅ Cleanup complete")