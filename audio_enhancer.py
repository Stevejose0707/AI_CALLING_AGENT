# cx_sip/audio_enhancer.py
import threading
import queue
import time
import numpy as np
import audioop
import torch
import torchaudio

# Optional: librosa for resampling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("   Librosa not available - using basic resampling")

class DeepFilterAudioEnhancer:
    """
    Threaded audio enhancer using DeepFilterNet via torch.hub.load.
    Supports real-time chunked processing with fallback to simple noise reduction.
    """
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processing = False
        self.input_queue = queue.Queue(maxsize=50)
        self.output_queue = queue.Queue(maxsize=50)
        self.worker_thread = None
        self.buffer = []
        self.buffer_size_ms = 80  # 80ms buffer
        self.fallback_mode = False

        # Load DeepFilterNet model via torch.hub
        try:
            self.model = torch.hub.load(
                'Rikorose/DeepFilterNet', 
                'deep_filter', 
                source='github'
            )
            self.model.to(self.device)
            self.model.eval()
            print("   ✅ DeepFilterNet model loaded successfully")
        except Exception as e:
            print(f"   ⚠️ Failed to load DeepFilterNet: {e}")
            self.model = None
            self.fallback_mode = True

    def start_processing(self):
        if not self.processing:
            self.processing = True
            self.worker_thread = threading.Thread(target=self._enhancement_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            mode = "DeepFilterNet" if self.model else "simple noise reduction"
            print(f"   🎛️ Enhancement started ({mode})")

    def _enhancement_loop(self):
        while self.processing:
            try:
                chunk_data = self.input_queue.get(timeout=0.1)
                ulaw_audio = chunk_data['audio']
                timestamp = chunk_data['timestamp']

                if self.model and not self.fallback_mode:
                    enhanced_audio = self._deepfilter_process_chunk(ulaw_audio)
                else:
                    enhanced_audio = self._simple_noise_reduction(ulaw_audio)

                if enhanced_audio is not None:
                    self.output_queue.put({
                        'audio': enhanced_audio,
                        'timestamp': timestamp,
                        'enhanced': True
                    })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"   Enhancement error: {e}")
                continue

    def _deepfilter_process_chunk(self, ulaw_audio):
        try:
            # μ-law to linear PCM
            linear_audio = audioop.ulaw2lin(ulaw_audio, 1)
            audio_array = np.frombuffer(linear_audio, dtype=np.int16).astype(np.float32)
            audio_normalized = audio_array / 32768.0  # [-1,1]

            # Resample to 48kHz
            if LIBROSA_AVAILABLE:
                audio_48k = librosa.resample(audio_normalized, orig_sr=8000, target_sr=48000)
            else:
                audio_48k = np.repeat(audio_normalized, 6)

            self.buffer.extend(audio_48k)
            required_samples = int(0.08 * 48000)  # 80ms
            if len(self.buffer) >= required_samples:
                chunk_to_process = np.array(self.buffer[:required_samples], dtype=np.float32)
                overlap_samples = int(0.02 * 48000)
                self.buffer = self.buffer[overlap_samples:]

                audio_input = torch.tensor(chunk_to_process).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_48k = self.model(audio_input).squeeze(0).cpu().numpy()

                # Downsample back to 8kHz
                if LIBROSA_AVAILABLE:
                    enhanced_8k = librosa.resample(enhanced_48k, orig_sr=48000, target_sr=8000)
                else:
                    enhanced_8k = enhanced_48k[::6]

                enhanced_int = np.clip(enhanced_8k * 32767, -32768, 32767).astype(np.int16)
                enhanced_ulaw = audioop.lin2ulaw(enhanced_int.tobytes(), 2)
                return enhanced_ulaw

        except Exception as e:
            print(f"   DeepFilter processing error: {e}")
            return self._simple_noise_reduction(ulaw_audio)

        return None

    def _simple_noise_reduction(self, ulaw_audio):
        """Fallback noise reduction"""
        try:
            linear_audio = audioop.ulaw2lin(ulaw_audio, 1)
            audio_array = np.frombuffer(linear_audio, dtype=np.int16).astype(np.float32)
            if len(audio_array) > 3:
                filtered = np.zeros_like(audio_array)
                filtered[1:] = audio_array[1:] - 0.95 * audio_array[:-1]
                threshold = np.max(np.abs(filtered)) * 0.1
                mask = np.abs(filtered) > threshold
                filtered = filtered * mask
                filtered_int = np.clip(filtered, -32768, 32767).astype(np.int16)
                enhanced_ulaw = audioop.lin2ulaw(filtered_int.tobytes(), 2)
                return enhanced_ulaw
        except Exception as e:
            print(f"   Simple noise reduction error: {e}")
        return ulaw_audio

    def queue_for_enhancement(self, ulaw_audio):
        if self.processing:
            try:
                self.input_queue.put_nowait({'audio': ulaw_audio, 'timestamp': time.time()})
            except queue.Full:
                pass

    def get_enhanced_audio(self):
        try:
            result = self.output_queue.get_nowait()
            return result['audio']
        except queue.Empty:
            return None

    def stop_processing(self):
        self.processing = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)

    def get_status(self):
        return {
            "deepfilter_available": self.model is not None,
            "processing": self.processing,
            "fallback_mode": self.fallback_mode,
            "queue_size": self.input_queue.qsize(),
            "output_ready": not self.output_queue.empty()
        }



