# cx_sip/sip_bridge.py
import socket
import os
import hashlib
import uuid
import re
import queue
from dotenv import load_dotenv

load_dotenv()

class SIPBridge:
    """Handles SIP protocol operations: registration, messaging, authentication"""
    
    def __init__(self):
        # Load configuration
        self.server = os.getenv("SIP_SERVER")
        self.auth_id = os.getenv("SIP_USERNAME")
        self.extension = os.getenv("SIP_EXTENSION")
        self.password = os.getenv("SIP_PASSWORD")
        self.domain = self.server
        
        # Get local IP
        self.local_ip = self._get_local_ip()
        
        # Socket setup
        self.sock = None
        self.local_port = None
        
        # Centralized SIP queue for socket reads
        self.sip_queue = queue.Queue()
        
        # State management
        self.registered = False
        self.running = True
        
        print("🔌 SIP Bridge Initialized")
        print(f"   Server: {self.server}")
        print(f"   Extension: {self.extension}")
        print(f"   Local IP: {self.local_ip}")
    
    def _get_local_ip(self):
        """Get local IP address"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self.server, 80))
            return s.getsockname()[0]
        finally:
            s.close()
    
    def setup_socket(self):
        """Setup UDP socket for SIP"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 0))
        self.local_port = self.sock.getsockname()[1]
        self.sock.settimeout(2.0)
        print(f"   SIP listening on port: {self.local_port}")
    
    def register(self):
        """Register with 3CX using correct headers"""
        print("\n📤 Registering with 3CX...")
        
        if not self.sock:
            self.setup_socket()
        
        call_id = str(uuid.uuid4())
        branch = f"z9hG4bK{uuid.uuid4().hex[:8]}"
        tag = uuid.uuid4().hex[:8]
        
        register = f"""REGISTER sip:{self.server}:5060 SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch={branch};rport
Max-Forwards: 70
Contact: <sip:{self.extension}@{self.local_ip}:{self.local_port};rinstance={uuid.uuid4().hex[:12]}>
To: "{self.extension}"<sip:{self.extension}@{self.server}:5060>
From: "{self.extension}"<sip:{self.extension}@{self.server}:5060>;tag={tag}
Call-ID: {call_id}
CSeq: 1 REGISTER
Expires: 120
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REGISTER, SUBSCRIBE, NOTIFY, REFER, INFO, MESSAGE
User-Agent: Python-SIP-Bridge/1.0
Content-Length: 0

""".replace('\n', '\r\n')
        
        self.sock.sendto(register.encode(), (self.server, 5060))
        
        try:
            data, addr = self.sock.recvfrom(65535)
            response = data.decode('utf-8', errors='ignore')
            
            if "407 Proxy Authentication Required" in response:
                print("   🔐 Authentication required...")
                
                match = re.search(r'Proxy-Authenticate: Digest (.+)\r\n', response)
                if match:
                    auth_params = match.group(1)
                    realm_match = re.search(r'realm="([^"]+)"', auth_params)
                    nonce_match = re.search(r'nonce="([^"]+)"', auth_params)
                    
                    if realm_match and nonce_match:
                        realm = realm_match.group(1)
                        nonce = nonce_match.group(1)
                        
                        # Calculate authentication
                        auth_header = self.calculate_auth(realm, nonce, "REGISTER", f"sip:{self.server}:5060")
                        
                        auth_register = f"""REGISTER sip:{self.server}:5060 SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch={branch}2;rport
Max-Forwards: 70
Contact: <sip:{self.extension}@{self.local_ip}:{self.local_port};rinstance={uuid.uuid4().hex[:12]}>
To: "{self.extension}"<sip:{self.extension}@{self.server}:5060>
From: "{self.extension}"<sip:{self.extension}@{self.server}:5060>;tag={tag}
Call-ID: {call_id}
CSeq: 2 REGISTER
Expires: 120
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REGISTER, SUBSCRIBE, NOTIFY, REFER, INFO, MESSAGE
Proxy-Authorization: {auth_header}
Supported: replaces
User-Agent: Python-SIP-Bridge/1.0
Content-Length: 0

""".replace('\n', '\r\n')
                        
                        self.sock.sendto(auth_register.encode(), (self.server, 5060))
                        
                        data, addr = self.sock.recvfrom(65535)
                        response = data.decode('utf-8', errors='ignore')
                        
                        if "200 OK" in response:
                            print("   ✅ Successfully registered with 3CX!")
                            self.registered = True
                            return True
                            
        except socket.timeout:
            print("   ⏱️ Registration timeout")
            return False
        except Exception as e:
            print(f"   ❌ Registration error: {e}")
            return False
    
    def calculate_auth(self, realm, nonce, method, uri):
        """Calculate MD5 digest authentication"""
        ha1 = hashlib.md5(f"{self.auth_id}:{realm}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        response_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
        
        return f'Digest username="{self.auth_id}",realm="{realm}",nonce="{nonce}",uri="{uri}",response="{response_hash}",algorithm=MD5'
    
    def send_sip_message(self, message, addr=None):
        """Send a SIP message"""
        if addr is None:
            addr = (self.server, 5060)
        self.sock.sendto(message.encode(), addr)
    
    def listen_for_messages(self):
        """Main loop to listen for SIP messages"""
        print("\n📞 SIP Bridge listening for messages...")
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                message = data.decode('utf-8', errors='ignore')
                
                try:
                    self.sip_queue.put_nowait((message, addr))
                except queue.Full:
                    pass
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"   Error (listener): {e}")
    
    def extract_header(self, message, header_name):
        """Extract a header value from SIP message"""
        pattern = f"{header_name}: (.+?)\\r\\n"
        match = re.search(pattern, message)
        return match.group(1) if match else ""
    
    def cleanup(self):
        """Clean shutdown"""
        self.running = False
        if self.sock:
            self.sock.close()