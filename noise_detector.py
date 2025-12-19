# cx_sip/noise_detector.py
import numpy as np
import audioop

class SimpleNoiseDetector:
    def __init__(self):
        self.noise_floor = None
        self.sample_count = 0
        
    def detect_noise(self, ulaw_audio_chunk):
        """
        Simple noise detection - returns True if noisy, False if clean
        """
        try:
            # Convert to linear audio
            linear_audio = audioop.ulaw2lin(ulaw_audio_chunk, 1)
            audio_array = np.frombuffer(linear_audio, dtype=np.int16)
            
            # Calculate energy
            energy = np.mean(audio_array.astype(np.float32) ** 2)
            
            # Learn noise floor for first 50 samples (1 second)
            if self.sample_count < 50:
                if self.noise_floor is None:
                    self.noise_floor = energy
                else:
                    self.noise_floor = 0.9 * self.noise_floor + 0.1 * energy
                self.sample_count += 1
                return False  # Assume clean during learning
            
            # Simple threshold: if energy is much higher than noise floor
            if self.noise_floor and energy > (self.noise_floor * 4):
                return False  # Clean speech (high energy)
            elif self.noise_floor and energy > (self.noise_floor * 2):
                return True   # Noisy
            else:
                return False  # Very quiet, assume clean
                
        except Exception:
            return False  # Default to clean on error