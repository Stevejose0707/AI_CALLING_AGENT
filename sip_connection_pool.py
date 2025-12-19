# src/cx_sip/sip_connection_pool.py

import queue
import threading
import time
from typing import Optional
from .sip_bridge import SIPBridge

class SIPConnectionPool:
    """Pool of pre-registered SIP connections for efficiency"""
    
    def __init__(self, pool_size: int = 20):
        self.pool_size = pool_size
        self.available_connections = queue.Queue()
        self.in_use_connections = set()
        self.all_connections = []
        self.lock = threading.Lock()
        self.running = True
        
        print(f"🏊 Initializing SIP connection pool (size: {pool_size})")
        
        # Initialize pool
        self._initialize_pool()
        
        # Start health check thread
        health_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True
        )
        health_thread.start()
    
    def _initialize_pool(self):
        """Create and register SIP connections"""
        successful = 0
        
        for i in range(self.pool_size):
            try:
                print(f"  📞 Creating connection {i+1}/{self.pool_size}...")
                bridge = SIPBridge()
                
                # Setup socket first
                bridge.setup_socket()
                
                # Try to register
                if bridge.register():
                    self.available_connections.put(bridge)
                    self.all_connections.append(bridge)
                    successful += 1
                    print(f"  ✅ Connection {i+1} registered")
                else:
                    print(f"  ❌ Connection {i+1} failed to register")
                    bridge.cleanup()
                    
            except Exception as e:
                print(f"  ❌ Connection {i+1} error: {e}")
        
        print(f"✅ SIP pool ready: {successful}/{self.pool_size} connections available")
    
    def get_connection(self, timeout: float = 30) -> Optional[SIPBridge]:
        """Get an available SIP connection from pool"""
        try:
            bridge = self.available_connections.get(timeout=timeout)
            
            with self.lock:
                self.in_use_connections.add(bridge)
            
            # Make sure it's still registered
            if not bridge.registered:
                print("  🔄 Re-registering stale connection...")
                if not bridge.register():
                    # Try to get another one
                    return self.get_connection(timeout=timeout/2)
            
            print(f"  ✅ Provided connection from pool (available: {self.available_connections.qsize()})")
            return bridge
            
        except queue.Empty:
            print("  ⚠️ No available SIP connections in pool")
            return None
    
    def return_connection(self, bridge: SIPBridge):
        """Return a connection to the pool"""
        if not bridge:
            return
        
        with self.lock:
            if bridge in self.in_use_connections:
                self.in_use_connections.remove(bridge)
        
        # Check if connection is still good
        if bridge.registered and bridge.sock:
            # Clear any pending messages in queue
            while not bridge.sip_queue.empty():
                try:
                    bridge.sip_queue.get_nowait()
                except:
                    break
            
            self.available_connections.put(bridge)
            print(f"  ♻️ Connection returned to pool (available: {self.available_connections.qsize() + 1})")
        else:
            print("  ⚠️ Connection not returned - not registered")
            # Try to fix it in the health check
    
    def _health_check_loop(self):
        """Periodically check and refresh connections"""
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            with self.lock:
                in_use_count = len(self.in_use_connections)
            
            available_count = self.available_connections.qsize()
            print(f"  📊 Pool status - Available: {available_count}, In use: {in_use_count}")
            
            # Re-register connections that need it
            for bridge in self.all_connections:
                if bridge not in self.in_use_connections:
                    if not bridge.registered:
                        try:
                            print("  🔄 Re-registering connection in health check")
                            bridge.register()
                        except Exception as e:
                            print(f"  ⚠️ Failed to re-register: {e}")
    
    def get_status(self):
        """Get pool status"""
        with self.lock:
            return {
                "total": len(self.all_connections),
                "available": self.available_connections.qsize(),
                "in_use": len(self.in_use_connections)
            }
    
    def shutdown(self):
        """Clean shutdown of all connections"""
        print("🛑 Shutting down SIP connection pool...")
        self.running = False
        
        # Clean up all connections
        for bridge in self.all_connections:
            try:
                bridge.cleanup()
            except:
                pass
        
        print("✅ SIP pool shutdown complete")

# Global SIP pool instance
_sip_pool = None
_pool_lock = threading.Lock()

def get_sip_pool() -> SIPConnectionPool:
    """Get or create the global SIP pool (singleton pattern)"""
    global _sip_pool
    
    if _sip_pool is None:
        with _pool_lock:
            if _sip_pool is None:  # Double-check
                _sip_pool = SIPConnectionPool(pool_size=20)
    
    return _sip_pool

def shutdown_pool():
    """Shutdown the global pool"""
    global _sip_pool
    if _sip_pool:
        _sip_pool.shutdown()
        _sip_pool = None