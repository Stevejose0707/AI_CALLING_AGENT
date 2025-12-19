# cx_sip/audio_path_manager.py
# cx_sip/audio_path_manager.py
import queue
from .noise_detector import SimpleNoiseDetector
from .audio_enhancer import DeepFilterAudioEnhancer

class AudioPathManager:
    def __init__(self, primary_queue):
        self.primary_queue = primary_queue
        self.noise_detector = SimpleNoiseDetector()
        
        # Statistics
        self.total_chunks = 0
        self.noisy_chunks = 0
        
        # Enhancement - THIS WAS MISSING
        self.audio_enhancer = DeepFilterAudioEnhancer()
        self.enhancement_started = False
        
    def route_audio(self, ulaw_audio):
        self.total_chunks += 1
        
        # Check for noise first
        is_noisy = self.noise_detector.detect_noise(ulaw_audio)
        
        # Get enhanced audio if available - THIS WAS MISSING
        enhanced_audio = None
        if self.enhancement_started:
            enhanced_audio = self.audio_enhancer.get_enhanced_audio()
        
        # Use enhanced audio if available, otherwise original - THIS WAS MISSING
        audio_to_use = enhanced_audio if enhanced_audio else ulaw_audio
        
        # Send to primary queue
        try:
            self.primary_queue.put_nowait(audio_to_use)
        except queue.Full:
            try:
                self.primary_queue.get_nowait()
                self.primary_queue.put_nowait(audio_to_use)
            except queue.Empty:
                pass
        
        # Handle noisy audio - THIS LOGIC WAS MISSING
        if is_noisy:
            self.noisy_chunks += 1
            
            if not self.enhancement_started:
                self.audio_enhancer.start_processing()
                self.enhancement_started = True
                print("   Started DeepFilter enhancement processing")
            
            self.audio_enhancer.queue_for_enhancement(ulaw_audio)
            
            if self.noisy_chunks % 10 == 0:
                status = "ENHANCED" if enhanced_audio else "PROCESSING"
                print(f"   Background noise detected - Status: {status}")
        
        # Log statistics
        if self.total_chunks % 250 == 0:
            noise_percent = (self.noisy_chunks / self.total_chunks) * 100
            enhancement_status = "ON" if self.enhancement_started else "OFF"
            print(f"   Audio stats: {noise_percent:.1f}% noisy, Enhancement: {enhancement_status}")