# cx_sip/rtp_handler.py - KEEP THIS VERSION
import socket
import os
import struct
import time 

class RTPHandler:
    """Handle RTP audio packets with minimal conversion"""
    def __init__(self, local_ip, local_port=None):
        self.local_ip = local_ip 
        self.remote_addr = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
        if local_port:
            self.sock.bind((local_ip, local_port))
            self.local_port = local_port
        else:
            self.sock.bind((local_ip, 0))  # 0 lets OS choose port
            self.local_port = self.sock.getsockname()[1]  # self.sock not sock
    
        self.sock.settimeout(0.02)
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = struct.unpack('!I', os.urandom(4))[0]
        self.running = True
    
    # NEW: Precise timing tracking
        self.playback_start_time = None
        self.total_audio_ms_sent = 0
        self.packet_send_times = {}  # Maps sequence numbers to send times
        self.bytes_to_ms_ratio = 8  # μ-law: 8 bytes per millisecond
    
        print(f"   🔻 RTP socket bound to {local_ip}:{self.local_port}")
        
    def parse_rtp_packet(self, data):
        """Extract audio from RTP packet"""
        if len(data) < 12:
            return None
        return data[12:]  # Just return raw audio
    
    def create_rtp_packet(self, audio_data):
        """Create RTP packet from audio data"""
        version = 2 << 6
        payload_type = 0  # PCMU
        marker = 0
        
        header = struct.pack('!BBHII',
            version | payload_type,
            marker,
            self.sequence_number,
            self.timestamp,
            self.ssrc
        )
        
        self.sequence_number = (self.sequence_number + 1) & 0xFFFF
        self.timestamp = (self.timestamp + 160) & 0xFFFFFFFF
        
        return header + audio_data
    
    def send_audio(self, audio_data):
        """Send audio via RTP with precise timing tracking"""
        if self.remote_addr and self.running:
            # Record timing for this packet
            send_time_ms = time.time() * 1000
            packet = self.create_rtp_packet(audio_data)
            # Track this packet's send time
            self.packet_send_times[self.sequence_number] = send_time_ms
            # Calculate audio duration (μ-law: 1 byte = 1/8 millisecond)
            audio_duration_ms = len(audio_data) / self.bytes_to_ms_ratio
            self.total_audio_ms_sent += audio_duration_ms
        
            try:
                self.sock.sendto(packet, self.remote_addr)
            except:
                pass
    def get_current_playback_position_ms(self):
        """Calculate exactly how many milliseconds of audio have been played"""
        if not self.playback_start_time:
            return 0
        # Time elapsed since playback started
        elapsed_time_ms = (time.time() - self.playback_start_time) * 1000
    
    # Account for network/buffer delays (typical telephony delay)
        buffer_delay_ms = 150
    
    # Calculate actual playback position
        actual_playback_ms = max(0, elapsed_time_ms - buffer_delay_ms)
    
    # Don't exceed total audio sent
        return min(actual_playback_ms, self.total_audio_ms_sent)

    def reset_timing(self):
        """Reset timing tracking for new audio stream"""
        self.playback_start_time = time.time()
        self.total_audio_ms_sent = 0
        self.packet_send_times.clear()
    def close(self):
        """Close the RTP socket"""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass