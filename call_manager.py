# cx_sip/call_manager.py - Complete version with DeepFilter and Voice Analysis

import time
import threading
import uuid
import re
import queue
import audioop
import asyncio
import json
import base64
import numpy as np
from .rtp_handler import RTPHandler
from .sip_bridge import SIPBridge
from .audio_enhancer import DeepFilterAudioEnhancer
from .audio_path_manager import AudioPathManager
from typing import AsyncIterator
import os
from datetime import datetime
from enum import Enum

# Import your existing OpenAI agent
from src.langchain_openai_voice import OpenAIVoiceReactAgent
from src.server.tools import TOOLS

from dotenv import load_dotenv
load_dotenv()

# Audio constants
FORMAT = 16
CHANNELS = 1
RATE = 24000
CHUNK = 512

class CallState(Enum):
    INIT = "initializing"
    ACTIVE = "active"
    ENDING = "ending"
    COMPLETED = "completed"

class CallStatus(Enum):
    INITIATING = "initiating"
    DIALING = "dialing"
    RINGING = "ringing"
    ANSWERED = "answered"
    BUSY = "busy"
    REJECTED = "rejected"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VoiceAnalyzer:
    """Real-time voice analysis for stress, deception, and emotion detection"""
    
    def __init__(self):
        self.baseline_pitch = None
        self.baseline_energy = None
        self.baseline_pause_pattern = []
        self.sample_count = 0
        self.analysis_history = []
        
        # Voice analysis parameters
        self.stress_indicators = {
            "pitch_variance": 0,
            "energy_spikes": 0,
            "pause_frequency": 0,
            "speech_rate": 0
        }
        
        # Deception detection features
        self.deception_markers = {
            "micro_hesitations": 0,
            "pitch_elevation": 0,
            "energy_inconsistency": 0,
            "vocal_tremor": 0
        }
        
        print("   🧠 Voice Analysis Engine initialized")
    
    def analyze_audio_chunk(self, ulaw_audio):
        """Analyze audio chunk for voice patterns"""
        try:
            # Convert μ-law to linear
            linear_audio = audioop.ulaw2lin(ulaw_audio, 1)
            audio_array = np.frombuffer(linear_audio, dtype=np.int16).astype(np.float32)
            
            if len(audio_array) < 160:  # Skip very short chunks
                return None
            
            # Calculate features
            energy = np.mean(audio_array ** 2)
            pitch_estimate = self._estimate_pitch(audio_array)
            zero_crossings = self._count_zero_crossings(audio_array)
            
            # Build baseline (first 5 seconds)
            if self.sample_count < 250:  # 5 seconds at 20ms chunks
                self._build_baseline(energy, pitch_estimate)
                self.sample_count += 1
                return {"status": "building_baseline", "progress": self.sample_count / 250}
            
            # Analyze against baseline
            analysis = self._analyze_against_baseline(energy, pitch_estimate, zero_crossings)
            self.analysis_history.append(analysis)
            
            # Keep only last 5 seconds of history
            if len(self.analysis_history) > 250:
                self.analysis_history = self.analysis_history[-250:]
            
            return self._generate_live_report()
            
        except Exception as e:
            print(f"   Voice analysis error: {e}")
            return None
    
    def _estimate_pitch(self, audio_array):
        """Simple pitch estimation using autocorrelation"""
        try:
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                normalized = audio_array / np.max(np.abs(audio_array))
            else:
                return 0
            
            # Autocorrelation
            correlation = np.correlate(normalized, normalized, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Find peak (avoiding zero lag)
            min_period = 20  # ~400Hz max
            max_period = 160  # ~50Hz min
            
            if len(correlation) > max_period:
                peak_idx = np.argmax(correlation[min_period:max_period]) + min_period
                if correlation[peak_idx] > 0.3:  # Confidence threshold
                    return 8000 / peak_idx  # Convert to Hz
            
            return 0
        except:
            return 0
    
    def _count_zero_crossings(self, audio_array):
        """Count zero crossings for speech analysis"""
        return np.sum(np.diff(np.signbit(audio_array)))
    
    def _build_baseline(self, energy, pitch):
        """Build baseline voice characteristics"""
        if self.baseline_energy is None:
            self.baseline_energy = energy
        else:
            self.baseline_energy = 0.9 * self.baseline_energy + 0.1 * energy
        
        if pitch > 0:
            if self.baseline_pitch is None:
                self.baseline_pitch = pitch
            else:
                self.baseline_pitch = 0.9 * self.baseline_pitch + 0.1 * pitch
    
    def _analyze_against_baseline(self, energy, pitch, zero_crossings):
        """Analyze current audio against baseline"""
        analysis = {
            "timestamp": time.time(),
            "energy_ratio": 0,
            "pitch_ratio": 0,
            "stress_level": 0,
            "deception_indicators": 0,
            "emotional_state": "neutral"
        }
        
        if self.baseline_energy and self.baseline_energy > 0:
            analysis["energy_ratio"] = energy / self.baseline_energy
        
        if self.baseline_pitch and self.baseline_pitch > 0 and pitch > 0:
            analysis["pitch_ratio"] = pitch / self.baseline_pitch
        
        # Stress detection
        stress_score = 0
        if analysis["energy_ratio"] > 1.5:  # Energy spike
            stress_score += 0.3
        if analysis["pitch_ratio"] > 1.2:  # Pitch elevation
            stress_score += 0.4
        if zero_crossings > 80:  # High speech rate
            stress_score += 0.3
        
        analysis["stress_level"] = min(stress_score, 1.0)
        
        # Deception indicators
        deception_score = 0
        if 1.15 < analysis["pitch_ratio"] < 1.4:  # Moderate pitch increase
            deception_score += 0.25
        if 0.5 < analysis["energy_ratio"] < 0.8:  # Reduced energy (hesitation)
            deception_score += 0.25
        if analysis["stress_level"] > 0.6:  # High stress
            deception_score += 0.5
        
        analysis["deception_indicators"] = min(deception_score, 1.0)
        
        # Emotional state classification
        if analysis["stress_level"] > 0.7:
            analysis["emotional_state"] = "stressed"
        elif analysis["deception_indicators"] > 0.6:
            analysis["emotional_state"] = "uncertain"
        elif analysis["energy_ratio"] > 1.3:
            analysis["emotional_state"] = "excited"
        elif analysis["energy_ratio"] < 0.7:
            analysis["emotional_state"] = "calm"
        
        return analysis
    
    def _generate_live_report(self):
        """Generate live analysis report"""
        if not self.analysis_history:
            return None
        
        # Average over last 2 seconds (100 chunks)
        recent = self.analysis_history[-100:] if len(self.analysis_history) >= 100 else self.analysis_history
        
        avg_stress = np.mean([a["stress_level"] for a in recent])
        avg_deception = np.mean([a["deception_indicators"] for a in recent])
        
        # Dominant emotional state
        emotions = [a["emotional_state"] for a in recent]
        dominant_emotion = max(set(emotions), key=emotions.count)
        
        return {
            "status": "analyzing",
            "stress_level": round(avg_stress, 2),
            "deception_probability": round(avg_deception, 2),
            "emotional_state": dominant_emotion,
            "confidence": min(len(recent) / 100, 1.0),
            "sample_count": len(self.analysis_history)
        }
    
    def get_current_analysis(self):
        """Get current voice analysis summary"""
        return self._generate_live_report()


class CallManager:
    """Enhanced CallManager with DeepFilter and Voice Analysis"""
    
    def __init__(self, custom_prompt=None, phone_number=None, greeting_message=None,
                call_context=None, enable_noise_reduction=True, enable_voice_analysis=True):
        
        # Initialize components
        self.sip_bridge = SIPBridge()
        self.rtp_handler = None
        
        # Configuration
        self.custom_prompt = custom_prompt
        self.greeting_message = greeting_message
        self.phone_number = phone_number
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_voice_analysis = enable_voice_analysis
        
        # Audio queues
        self.q_in = queue.Queue()
        self.q_out = queue.Queue()
        self.input_buffer = bytearray()
        self.chunk_size = 4800
        
        # Audio enhancement components
        if self.enable_noise_reduction:
            self.audio_enhancer = DeepFilterAudioEnhancer()
            self.audio_path_manager = AudioPathManager(self.q_in)
            print("   🔧 DeepFilter noise cancellation enabled")
        else:
            self.audio_enhancer = None
            self.audio_path_manager = None
            print("   ⚠️ DeepFilter disabled")
        
        # Voice analysis component
        if self.enable_voice_analysis:
            self.voice_analyzer = VoiceAnalyzer()
            print("   🧠 Voice analysis enabled")
        else:
            self.voice_analyzer = None
        
        # Enhanced audio queue for processed audio
        self.enhanced_audio_queue = queue.Queue()
        
        # Call state
        self.active_calls = {}
        self.current_outbound_call_id = None
        self.running = True
        
        # State Machine
        self.call_state = CallState.INIT
        self.state_lock = threading.Lock()
        self.hangup_lock = threading.Lock()
        self.hangup_initiated = False
        self.hangup_event = threading.Event()
        
        # Call context and metadata
        self.call_context = call_context or {}
        self.call_metadata = {
            "status": "pending",
            "use_case": self.call_context.get("use_case", "generic"),
            "phone_number": phone_number,
            "deepfilter_enabled": self.enable_noise_reduction,
            "voice_analysis_enabled": self.enable_voice_analysis
        }
        
        # Conversation tracking
        self.conversation_transcript = []
        
        # OpenAI agent components
        self.openai_agent = None
        self.agent_task = None
        self.agent_loop = None
        self.call_start_time = None
        
        # Audio tracking
        self.current_assistant_item_id = None
        self.audio_bytes_sent_to_rtp = 0
        self.current_audio_start_ms = 0
        self.playback_start_time = None
        self.bytes_to_ms_ratio = 8
        
        # Statistics and analysis
        self.audio_stats = {
            "total_chunks": 0,
            "enhanced_chunks": 0,
            "noise_detected": 0,
            "voice_analysis_samples": 0
        }
        
        self.live_voice_analysis = {}
        self.enhancement_status = {
            "deepfilter_working": False,
            "last_update": None,
            "processed_count": 0
        }
        
        features_enabled = []
        if self.enable_noise_reduction:
            features_enabled.append("DeepFilter")
        if self.enable_voice_analysis:
            features_enabled.append("Voice Analysis")
        
        print(f"🚀 Enhanced CallManager: {', '.join(features_enabled) if features_enabled else 'Basic Mode'}")
    
    def start(self):
        """Start the enhanced call manager"""
        if not self.sip_bridge.register():
            print("❌ Failed to register with SIP server")
            return False
        
        listen_thread = threading.Thread(target=self.sip_bridge.listen_for_messages)
        listen_thread.daemon = True
        listen_thread.start()
        
        refresh_thread = threading.Thread(target=self.keep_registration_alive)
        refresh_thread.daemon = True
        refresh_thread.start()
        
        # Start audio enhancement if enabled
        if self.audio_enhancer:
            self.audio_enhancer.start_processing()
            enhancement_thread = threading.Thread(target=self._monitor_enhanced_audio)
            enhancement_thread.daemon = True
            enhancement_thread.start()
            
            # Start enhancement status monitor
            status_thread = threading.Thread(target=self._monitor_enhancement_status)
            status_thread.daemon = True
            status_thread.start()
        
        return True
    
    def _monitor_enhanced_audio(self):
        """Monitor for enhanced audio from DeepFilter"""
        print("   🔧 Audio enhancement monitor started")
        
        while self.running:
            try:
                enhanced_audio = self.audio_enhancer.get_enhanced_audio()
                if enhanced_audio:
                    # Queue the enhanced audio for OpenAI
                    try:
                        self.enhanced_audio_queue.put_nowait(enhanced_audio)
                        self.audio_stats["enhanced_chunks"] += 1
                        self.enhancement_status["processed_count"] += 1
                        self.enhancement_status["deepfilter_working"] = True
                        self.enhancement_status["last_update"] = datetime.now().isoformat()
                    except queue.Full:
                        # Replace oldest with newest if queue is full
                        try:
                            self.enhanced_audio_queue.get_nowait()
                            self.enhanced_audio_queue.put_nowait(enhanced_audio)
                        except queue.Empty:
                            pass
                
                time.sleep(0.01)  # 10ms check interval
                
            except Exception as e:
                if self.running:
                    print(f"   Audio enhancement monitor error: {e}")
                break
    
    def _monitor_enhancement_status(self):
        """Monitor and report enhancement status"""
        last_processed_count = 0
        
        while self.running:
            try:
                time.sleep(5)  # Check every 5 seconds
                
                current_count = self.enhancement_status["processed_count"]
                if current_count > last_processed_count:
                    print(f"   ✅ DeepFilter processing: {current_count - last_processed_count} chunks/5s")
                    last_processed_count = current_count
                else:
                    if self.enhancement_status["deepfilter_working"]:
                        print("   ⚠️ DeepFilter may have stopped processing")
                        self.enhancement_status["deepfilter_working"] = False
                
                # Voice analysis status
                if self.voice_analyzer:
                    analysis = self.voice_analyzer.get_current_analysis()
                    if analysis:
                        self.live_voice_analysis = analysis
                        if analysis["status"] == "analyzing":
                            stress = analysis["stress_level"]
                            deception = analysis["deception_probability"]
                            emotion = analysis["emotional_state"]
                            print(f"   🧠 Voice Analysis: Stress={stress:.2f}, Deception={deception:.2f}, Emotion={emotion}")
                
            except Exception as e:
                if self.running:
                    print(f"   Status monitor error: {e}")
    
    def start_openai_connection(self):
        """Start OpenAI agent when call connects"""
        print("   🎤 Starting OpenAI Voice Agent with enhancements...")
        self.running = True
        self.call_start_time = time.time()
        
        # Create the agent thread
        self.agent_thread = threading.Thread(target=self._run_openai_agent)
        self.agent_thread.daemon = True
        self.agent_thread.start()
        
        time.sleep(1)
        print("   ✅ OpenAI Agent connected with enhancements")
    
    def _run_openai_agent(self):
        """Run OpenAI agent in thread with asyncio"""
        self.agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.agent_loop)
        
        try:
            self.agent_loop.run_until_complete(self._connect_openai_agent())
        finally:
            self.agent_loop.close()
            self.agent_loop = None
    
    async def _connect_openai_agent(self):
        """Connect OpenAI agent with enhancements"""
        agent_task = None
        sip_monitor_task = None
        
        try:
            # Create enhanced input stream
            input_stream = self._create_enhanced_input_stream()
            
            # Callbacks for agent events
            def handle_silence_timeout(reason: str, notes: str):
                if self.running:
                    print(f"   🔴 Silence timeout: {reason}")
                    asyncio.create_task(self._trigger_unified_hangup(reason, notes))
            
            async def handle_end_call(reason: str, notes: str):
                if self.running:
                    print(f"   🎯 Agent ending call: {reason}")
                    await self._trigger_unified_hangup(f"agent_{reason}", notes)
            
            # Create OpenAI agent with enhanced capabilities
            self.openai_agent = OpenAIVoiceReactAgent(
                model="gpt-4o-mini-realtime-preview",
                tools=TOOLS,
                instructions=self.custom_prompt,
                greeting_message=self.greeting_message,
                on_end_call_callback=handle_end_call,
                on_silence_timeout_callback=handle_silence_timeout,
                # Optimized VAD settings for enhanced audio
                vad_threshold=0.5 if self.enable_noise_reduction else 0.6,
                silence_duration_ms=300 if self.enable_noise_reduction else 400,
                prefix_padding_ms=100 if self.enable_noise_reduction else 150,
                noise_reduction_type="far_field" if self.enable_noise_reduction else "near_field"
            )
            
            # Create enhanced output handler
            def create_enhanced_output_handler():
                async def send_output(data: str):
                    if not self.running:
                        return
                    
                    try:
                        msg = json.loads(data) if isinstance(data, str) else data
                        msg_type = msg.get("type", "")
                        
                        # Handle audio output
                        if msg_type == "response.audio.delta":
                            if self.running:
                                audio_base64 = msg.get("delta", "")
                                ulaw_bytes = base64.b64decode(audio_base64)
                                self.q_out.put(ulaw_bytes)
                        
                        # Enhanced interruption detection
                        elif msg_type == "input_audio_buffer.speech_started":
                            print("   🛑 User interruption detected (enhanced processing)")
                            ms_played = self.audio_bytes_sent_to_rtp // self.bytes_to_ms_ratio
                            print(f"   📊 Audio played before interrupt: {ms_played}ms")
                            
                            # Clear output queue
                            while not self.q_out.empty():
                                try:
                                    self.q_out.get_nowait()
                                except:
                                    break
                            
                            print("   ✅ Audio queue cleared - enhanced interruption processed")
                        
                        # Handle transcripts with voice analysis
                        elif msg_type == "assistant_transcript":
                            transcript = msg.get("transcript", "")
                            if transcript:
                                self.conversation_transcript.append({
                                    "speaker": "agent",
                                    "text": transcript,
                                    "timestamp": datetime.now().isoformat(),
                                    "voice_analysis": self.live_voice_analysis.copy() if self.live_voice_analysis else None
                                })
                                print(f"   🤖 Assistant: {transcript}")
                        
                        elif msg_type == "user_transcript":
                            transcript = msg.get("transcript", "")
                            if transcript:
                                # Include voice analysis in transcript
                                transcript_entry = {
                                    "speaker": "user",
                                    "text": transcript,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                # Add voice analysis if available
                                if self.live_voice_analysis:
                                    transcript_entry["voice_analysis"] = self.live_voice_analysis.copy()
                                
                                self.conversation_transcript.append(transcript_entry)
                                print(f"   👤 User: {transcript}")
                                
                                # Print voice analysis if interesting
                                if self.live_voice_analysis and self.live_voice_analysis.get("status") == "analyzing":
                                    stress = self.live_voice_analysis.get("stress_level", 0)
                                    deception = self.live_voice_analysis.get("deception_probability", 0)
                                    if stress > 0.6 or deception > 0.5:
                                        print(f"   🧠 Notable: Stress={stress:.2f}, Deception={deception:.2f}")
                        
                        # Handle hangup signals
                        elif msg_type in ["silence_timeout_hangup", "agent_end_call_hangup"]:
                            reason = msg.get("reason", "unknown")
                            notes = msg.get("notes", "")
                            await self._trigger_unified_hangup(reason, notes)
                    
                    except Exception as e:
                        if self.running:
                            print(f"   Output handler error: {e}")
                
                return send_output
            
            # Start agent task
            agent_task = asyncio.create_task(
                self.openai_agent.aconnect(input_stream, create_enhanced_output_handler())
            )
            
            # Start SIP monitor
            sip_monitor_task = asyncio.create_task(self._monitor_sip_messages())
            
            # Wait for tasks to complete
            done, pending = await asyncio.wait(
                [agent_task, sip_monitor_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        except asyncio.CancelledError:
            print("   ⚠️ Agent task cancelled")
        except Exception as e:
            print(f"   ❌ OpenAI agent error: {e}")
            if self.running:
                await self._trigger_unified_hangup("agent_error", str(e))
        finally:
            # Proper cleanup
            if agent_task and not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            
            if sip_monitor_task and not sip_monitor_task.done():
                sip_monitor_task.cancel()
                try:
                    await sip_monitor_task
                except asyncio.CancelledError:
                    pass
    
    async def _create_enhanced_input_stream(self) -> AsyncIterator[str]:
        """Create enhanced input stream with DeepFilter priority"""
        try:
            while self.running:
                with self.state_lock:
                    if self.call_state == CallState.COMPLETED:
                        break
                    if self.call_state == CallState.ENDING:
                        break
                    if self.call_state != CallState.ACTIVE:
                        await asyncio.sleep(0.1)
                        continue
                
                audio_to_send = None
                audio_source = "raw"
                
                # Priority 1: Enhanced audio if available
                if self.audio_enhancer and not self.enhanced_audio_queue.empty():
                    try:
                        audio_to_send = self.enhanced_audio_queue.get_nowait()
                        audio_source = "enhanced"
                    except queue.Empty:
                        pass
                
                # Priority 2: Raw audio if no enhanced available
                if audio_to_send is None and not self.q_in.empty():
                    try:
                        audio_to_send = self.q_in.get_nowait()
                        audio_source = "raw"
                    except queue.Empty:
                        pass
                
                if audio_to_send:
                    # Send audio to OpenAI
                    base64_audio = base64.b64encode(audio_to_send).decode('utf-8')
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": base64_audio
                    }
                    
                    # Log enhancement usage every 100 chunks
                    if self.audio_stats["total_chunks"] % 100 == 0 and audio_source == "enhanced":
                        print(f"   🔧 Using {audio_source} audio (chunk {self.audio_stats['total_chunks']})")
                    
                    yield json.dumps(message)
                
                await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"   Enhanced input stream error: {e}")
        finally:
            print("   Enhanced input stream ending")
    
    def handle_rtp_receive(self, call_id):
        """Enhanced RTP receive with voice analysis"""
        print("   🔥 Enhanced RTP audio processing started...")
        
        while call_id in self.active_calls and self.rtp_handler and self.rtp_handler.running:
            with self.state_lock:
                if self.call_state != CallState.ACTIVE:
                    break
            
            try:
                data, addr = self.rtp_handler.sock.recvfrom(4096)
                ulaw_audio = self.rtp_handler.parse_rtp_packet(data)
                
                if ulaw_audio and len(ulaw_audio) > 0:
                    self.audio_stats["total_chunks"] += 1
                    
                    with self.state_lock:
                        if self.call_state == CallState.ACTIVE:
                            # Process through audio path manager
                            if self.audio_path_manager:
                                self.audio_path_manager.route_audio(ulaw_audio)
                                
                                # Send to enhancer for processing
                                if self.audio_enhancer:
                                    self.audio_enhancer.queue_for_enhancement(ulaw_audio)
                            else:
                                # Fallback to direct queuing
                                self.q_in.put(ulaw_audio)
                            
                            # Voice analysis
                            if self.voice_analyzer:
                                analysis = self.voice_analyzer.analyze_audio_chunk(ulaw_audio)
                                if analysis:
                                    self.audio_stats["voice_analysis_samples"] += 1
                                    if analysis.get("status") == "analyzing":
                                        self.live_voice_analysis = analysis
                                        
            except Exception as e:
                if call_id in self.active_calls:
                    continue
                break
    
    def play_audio_from_queue(self, call_id):
        """Enhanced audio playback to phone"""
        print("   📤 Enhanced audio playback started...")
        
        was_empty = False
        delay_applied = False
        
        # Reset tracking
        self.audio_bytes_sent_to_rtp = 0
        self.playback_start_time = time.time()
        
        while call_id in self.active_calls and self.rtp_handler and self.rtp_handler.running:
            with self.state_lock:
                if self.call_state != CallState.ACTIVE:
                    break
            
            if self.hangup_event.is_set():
                break
            
            try:
                if self.q_out.empty():
                    was_empty = True
                    delay_applied = False
                    time.sleep(0.001)
                    continue
                
                # Enhanced post-interruption handling
                if was_empty and not delay_applied:
                    print("   🎵 Brief pause for smooth recovery")
                    
                    # Reset tracking
                    self.audio_bytes_sent_to_rtp = 0
                    self.playback_start_time = time.time()
                    
                    # Brief silence with adaptive delay
                    silence_packet = b'\xff' * 160
                    delay_ms = 200 if self.enable_noise_reduction else 300  # Shorter with clean audio
                    num_packets = delay_ms // 20
                    for _ in range(num_packets):
                        self.rtp_handler.send_audio(silence_packet)
                        time.sleep(0.020)
                    
                    was_empty = False
                    delay_applied = True
                
                if not self.q_out.empty():
                    audio_chunks = []
                    
                    # Collect available chunks
                    while not self.q_out.empty() and len(audio_chunks) < 10:
                        try:
                            audio_chunk = self.q_out.get_nowait()
                            audio_chunks.append(audio_chunk)
                        except queue.Empty:
                            break
                    
                    if audio_chunks:
                        # Combine μ-law chunks
                        combined_ulaw = b''.join(audio_chunks)
                        
                        # Send via RTP with tracking
                        for i in range(0, len(combined_ulaw), 160):
                            chunk = combined_ulaw[i:i+160]
                            if len(chunk) < 160:
                                chunk = chunk + (b'\xff' * (160 - len(chunk)))
                            
                            self.rtp_handler.send_audio(chunk)
                            self.audio_bytes_sent_to_rtp += len(chunk)
                            time.sleep(0.020)
                else:
                    time.sleep(0.001)
                        
            except Exception as e:
                if call_id in self.active_calls:
                    print(f"   Audio error: {e}")
                break
    
    async def _trigger_unified_hangup(self, reason: str, notes: str = ""):
        """Enhanced unified hangup handler"""
        with self.hangup_lock:
            if self.hangup_initiated:
                return
            self.hangup_initiated = True
        
        print(f"\n🎯 HANGUP TRIGGERED: {reason}")
        
        # Stop everything immediately
        self.running = False
        
        if not self.set_state(CallState.ENDING):
            return
        
        self.hangup_event.set()
        
        # Stop enhancements
        if self.audio_enhancer:
            self.audio_enhancer.stop_processing()
            print("   🔧 Audio enhancement stopped")
        
        # Clear all queues
        while not self.q_out.empty():
            try: self.q_out.get_nowait()
            except: break
            
        while not self.q_in.empty():
            try: self.q_in.get_nowait()
            except: break
            
        while not self.enhanced_audio_queue.empty():
            try: self.enhanced_audio_queue.get_nowait()
            except: break
        
        # Prepare enhanced final metadata
        duration = time.time() - self.call_start_time if self.call_start_time else 0
        
        # Calculate enhancement statistics
        enhancement_stats = {}
        if self.enable_noise_reduction and self.audio_stats["total_chunks"] > 0:
            enhancement_rate = (self.audio_stats["enhanced_chunks"] / self.audio_stats["total_chunks"]) * 100
            enhancement_stats = {
                "deepfilter_enabled": True,
                "enhancement_rate": round(enhancement_rate, 2),
                "total_chunks": self.audio_stats["total_chunks"],
                "enhanced_chunks": self.audio_stats["enhanced_chunks"],
                "deepfilter_working": self.enhancement_status["deepfilter_working"]
            }
        
        # Voice analysis summary
        voice_analysis_summary = {}
        if self.enable_voice_analysis and self.live_voice_analysis:
            voice_analysis_summary = {
                "voice_analysis_enabled": True,
                "final_analysis": self.live_voice_analysis.copy(),
                "total_samples": self.audio_stats["voice_analysis_samples"]
            }
        
        self.final_call_state = {
            "call_id": self.current_outbound_call_id,
            "was_initiated": bool(self.current_outbound_call_id),
            "phone_number": self.phone_number,
            "use_case": self.call_metadata.get("use_case"),
            "duration_seconds": round(duration, 2),
            "end_reason": reason,
            "call_disconnected": "user_hangup" if "user_hangup" in reason else "agent_hangup",
            "total_messages": len(self.conversation_transcript),
            "conversation_text": self.conversation_transcript,
            "call_context": self.call_context,
            "call_status": "completed",
            "enhancements": {
                **enhancement_stats,
                **voice_analysis_summary
            }
        }
        
        # Print enhancement summary
        if enhancement_stats:
            print(f"   📊 DeepFilter: {enhancement_stats['enhancement_rate']:.1f}% enhanced")
        if voice_analysis_summary and voice_analysis_summary.get("final_analysis"):
            final = voice_analysis_summary["final_analysis"]
            print(f"   🧠 Voice Analysis: Stress={final.get('stress_level', 0):.2f}, " +
                  f"Deception={final.get('deception_probability', 0):.2f}")
        
        self.final_conversation = self.conversation_transcript.copy()
        
        # Do SIP hangup
        if self.current_outbound_call_id:
            self.hangup_call()
        
        self.set_state(CallState.COMPLETED)
    
    async def _monitor_sip_messages(self):
        """Monitor for SIP BYE messages"""
        while self.running:
            try:
                if not self.sip_bridge.sip_queue.empty():
                    msg, addr = self.sip_bridge.sip_queue.get_nowait()
                    
                    if "BYE" in msg and self.current_outbound_call_id:
                        call_id_in_msg = self.sip_bridge.extract_header(msg, "Call-ID")
                        if call_id_in_msg == self.current_outbound_call_id:
                            print("   📞 User hung up - BYE received")
                            
                            # Send 200 OK response
                            ok_response = f"""SIP/2.0 200 OK
Via: {self.sip_bridge.extract_header(msg, "Via")}
From: {self.sip_bridge.extract_header(msg, "From")}
To: {self.sip_bridge.extract_header(msg, "To")}
Call-ID: {call_id_in_msg}
CSeq: {self.sip_bridge.extract_header(msg, "CSeq")}
Content-Length: 0

""".replace('\n', '\r\n')
                            self.sip_bridge.send_sip_message(ok_response, addr)
                            
                            # Trigger hangup
                            await self._trigger_unified_hangup("user_hangup", "User ended call")
                            return
                        else:
                            self.sip_bridge.sip_queue.put_nowait((msg, addr))
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                if self.running:
                    print(f"   SIP monitor error: {e}")
                break
    
    def set_state(self, new_state: CallState) -> bool:
        """Thread-safe state transition"""
        with self.state_lock:
            valid_transitions = {
                CallState.INIT: [CallState.ACTIVE],
                CallState.ACTIVE: [CallState.ENDING],
                CallState.ENDING: [CallState.COMPLETED],
                CallState.COMPLETED: []
            }
            
            if new_state in valid_transitions[self.call_state]:
                print(f"📊 State: {self.call_state.value} → {new_state.value}")
                self.call_state = new_state
                return True
            return False
    
    def get_live_status(self):
        """Get live status of all enhancements"""
        status = {
            "call_active": self.call_state == CallState.ACTIVE,
            "call_duration": time.time() - self.call_start_time if self.call_start_time else 0,
            "audio_stats": self.audio_stats.copy(),
            "deepfilter_status": None,
            "voice_analysis": None
        }
        
        # DeepFilter status
        if self.audio_enhancer:
            status["deepfilter_status"] = {
                **self.enhancement_status,
                "enhancer_status": self.audio_enhancer.get_status()
            }
        
        # Voice analysis status
        if self.voice_analyzer and self.live_voice_analysis:
            status["voice_analysis"] = self.live_voice_analysis.copy()
        
        return status
    
    def make_outbound_call(self, destination=None):
        """Initiate enhanced outbound call"""
        if destination is None:
            destination = self.phone_number
        
        if not destination:
            print("❌ No phone number provided")
            return False
        
        features = []
        if self.enable_noise_reduction:
            features.append("DeepFilter")
        if self.enable_voice_analysis:
            features.append("Voice Analysis")
        
        feature_str = f" with {', '.join(features)}" if features else ""
        print(f"\n📞 Initiating enhanced call to {destination}{feature_str}...")
        
        if not self.sip_bridge.registered:
            print("❌ Not registered. Please register first.")
            return False
        
        self.set_state(CallState.ACTIVE)
        self.call_status = CallStatus.DIALING
        
        # Generate call identifiers
        call_id = str(uuid.uuid4())
        branch = f"z9hG4bK{uuid.uuid4().hex[:8]}"
        tag = uuid.uuid4().hex[:8]
        
        # Set enhanced metadata
        self.call_metadata.update({
            "call_id": call_id,
            "call_initiated_at": datetime.now().isoformat(),
            "initial_status": "initiated",
            "enhancements_enabled": {
                "deepfilter": self.enable_noise_reduction,
                "voice_analysis": self.enable_voice_analysis
            }
        })
        
        # Continue with RTP setup
        self.rtp_handler = RTPHandler(self.sip_bridge.local_ip)
        
        sdp = f"""v=0
o=- {int(time.time())} {int(time.time())} IN IP4 {self.sip_bridge.local_ip}
s=Python SIP Session
c=IN IP4 {self.sip_bridge.local_ip}
t=0 0
m=audio {self.rtp_handler.local_port} RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv"""
                    
        invite = f"""INVITE sip:{destination}@{self.sip_bridge.server}:5060 SIP/2.0
Via: SIP/2.0/UDP {self.sip_bridge.local_ip}:{self.sip_bridge.local_port};branch={branch};rport
Max-Forwards: 70
Contact: <sip:{self.sip_bridge.extension}@{self.sip_bridge.local_ip}:{self.sip_bridge.local_port}>
To: <sip:{destination}@{self.sip_bridge.server}:5060>
From: "{self.sip_bridge.extension}"<sip:{self.sip_bridge.extension}@{self.sip_bridge.server}:5060>;tag={tag}
Call-ID: {call_id}
CSeq: 1 INVITE
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REGISTER, SUBSCRIBE, NOTIFY, REFER, INFO, MESSAGE
Content-Type: application/sdp
Supported: replaces
User-Agent: Python-SIP-Bridge/1.0
Content-Length: {len(sdp)}

{sdp}""".replace('\n', '\r\n')
        
        self.sip_bridge.send_sip_message(invite)
        print("   ➡️ Sent: INVITE")
        
        return self.handle_call_setup(call_id, tag, destination, sdp)
    
    def handle_call_setup(self, call_id, tag, destination, sdp):
        """Handle call setup responses"""
        auth_required = False
        nonce = None
        realm = None
        start_time = time.time()
        overall_timeout = 30.0
        
        try:
            while time.time() - start_time < overall_timeout:
                try:
                    msg, addr = self.sip_bridge.sip_queue.get(timeout=2.0)
                except queue.Empty:
                    continue
                
                response = msg
                first_line = response.split('\r\n')[0] if response else '<empty>'
                print(f"   ← SIP from {addr}: {first_line}")
                
                resp_call_id = self.sip_bridge.extract_header(response, "Call-ID")
                if resp_call_id and resp_call_id != call_id:
                    try:
                        self.sip_bridge.sip_queue.put_nowait((response, addr))
                    except queue.Full:
                        pass
                    continue
                
                if "407 Proxy Authentication Required" in response:
                    print("   🔐 Authentication required...")
                    auth_required = True
                    
                    match = re.search(r'Proxy-Authenticate: Digest (.+)\r\n', response)
                    if match:
                        auth_params = match.group(1)
                        realm_match = re.search(r'realm="([^"]+)"', auth_params)
                        nonce_match = re.search(r'nonce="([^"]+)"', auth_params)
                        
                        if realm_match and nonce_match:
                            realm = realm_match.group(1)
                            nonce = nonce_match.group(1)
                    
                    self.send_ack(call_id, tag, destination, response)
                    break
                    
                elif " 100 " in response:
                    print("   ➡️ Received: 100 Trying")
                    self.call_status = CallStatus.DIALING
                    
                elif " 180 " in response or " 183 " in response:
                    print("   ➡️ Received: 180/183 Ringing")
                    self.call_status = CallStatus.RINGING
                    
                elif " 200 OK" in response:
                    print("   ✅ Call answered!")
                    self.call_status = CallStatus.ANSWERED
                    return self.handle_call_connected(call_id, tag, destination, response, addr)
                    
                elif " 486 " in response or " 603 " in response:
                    print("   ❌ Call rejected/busy")
                    self.call_status = CallStatus.BUSY
                    self.send_ack(call_id, tag, destination, response)
                    return False
                    
        except Exception as e:
            print(f"   Error during call setup: {e}")
            self.call_status = CallStatus.FAILED
            return False
        
        if auth_required and nonce and realm:
            return self.send_authenticated_invite(call_id, tag, destination, sdp, realm, nonce)
        
        self.call_status = CallStatus.NO_ANSWER
        return False
    
    def send_authenticated_invite(self, call_id, tag, destination, sdp, realm, nonce):
        """Send INVITE with authentication"""
        uri = f"sip:{destination}@{self.sip_bridge.server}:5060"
        auth_header = self.sip_bridge.calculate_auth(realm, nonce, "INVITE", uri)
        
        branch = f"z9hG4bK{uuid.uuid4().hex[:8]}"
        
        auth_invite = f"""INVITE sip:{destination}@{self.sip_bridge.server}:5060 SIP/2.0
Via: SIP/2.0/UDP {self.sip_bridge.local_ip}:{self.sip_bridge.local_port};branch={branch};rport
Max-Forwards: 70
Contact: <sip:{self.sip_bridge.extension}@{self.sip_bridge.local_ip}:{self.sip_bridge.local_port}>
To: <sip:{destination}@{self.sip_bridge.server}:5060>
From: "{self.sip_bridge.extension}"<sip:{self.sip_bridge.extension}@{self.sip_bridge.server}:5060>;tag={tag}
Call-ID: {call_id}
CSeq: 2 INVITE
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REGISTER, SUBSCRIBE, NOTIFY, REFER, INFO, MESSAGE
Proxy-Authorization: {auth_header}
Content-Type: application/sdp
Supported: replaces
User-Agent: Python-SIP-Bridge/1.0
Content-Length: {len(sdp)}

{sdp}""".replace('\n', '\r\n')
        
        self.sip_bridge.send_sip_message(auth_invite)
        print("   ➡️ Sent: Authenticated INVITE")
        
        return self.wait_for_authenticated_response(call_id, tag, destination)
    
    def wait_for_authenticated_response(self, call_id, tag, destination):
        """Wait for response after authenticated INVITE"""
        start_time = time.time()
        timeout = 30.0
        
        while time.time() - start_time < timeout:
            try:
                msg, addr = self.sip_bridge.sip_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            
            response = msg
            
            resp_call_id = self.sip_bridge.extract_header(response, "Call-ID")
            if resp_call_id and resp_call_id != call_id:
                try:
                    self.sip_bridge.sip_queue.put_nowait((response, addr))
                except queue.Full:
                    pass
                continue
            
            if " 200 OK" in response:
                print("   ✅ Call answered!")
                return self.handle_call_connected(call_id, tag, destination, response, addr)
            elif " 486 " in response or " 603 " in response:
                print("   ❌ Call rejected/busy")
                self.send_ack(call_id, tag, destination, response)
                return False
        
        print("   ⏱️ Call setup timeout")
        return False
    
    def handle_call_connected(self, call_id, tag, destination, response, addr):
        """Handle successful call connection with enhancements"""
        remote_rtp_port = 10000
        if "m=audio" in response:
            m_line = re.search(r'm=audio (\d+)', response)
            if m_line:
                remote_rtp_port = int(m_line.group(1))
        
        remote_ip = addr[0]
        c_line = re.search(r'c=IN IP4 ([\d.]+)', response)
        if c_line:
            remote_ip = c_line.group(1)
        
        self.rtp_handler.remote_addr = (remote_ip, remote_rtp_port)
        print(f"   📡 RTP configured: {self.sip_bridge.local_ip}:10000 ⟷ {remote_ip}:{remote_rtp_port}")
        
        self.send_ack(call_id, tag, destination, response)
        
        to_tag = ""
        to_match = re.search(r'To:.*?tag=([^;\r\n\s]+)', response)
        if to_match:
            to_tag = f";tag={to_match.group(1)}"
        
        self.active_calls[call_id] = {
            'from': f'"{self.sip_bridge.extension}"<sip:{self.sip_bridge.extension}@{self.sip_bridge.server}:5060>;tag={tag}',
            'to': f'<sip:{destination}@{self.sip_bridge.server}:5060>{to_tag}',
            'addr': addr,
            'remote_rtp': (remote_ip, remote_rtp_port),
            'start_time': time.time(),
            'direction': 'outbound',
            'destination': destination,
            'enhancements_active': {
                'deepfilter': self.enable_noise_reduction,
                'voice_analysis': self.enable_voice_analysis
            }
        }
        
        self.start_openai_connection()
        
        self.call_metadata.update({
            "connected_at": datetime.now().isoformat(),
            "connection_status": "connected"
        })
        
        rtp_receive_thread = threading.Thread(
            target=self.handle_rtp_receive,
            args=(call_id,)
        )
        rtp_receive_thread.daemon = True
        rtp_receive_thread.start()
        
        openai_output_thread = threading.Thread(
            target=self.play_audio_from_queue,
            args=(call_id,)
        )
        openai_output_thread.daemon = True
        openai_output_thread.start()
        
        features = []
        if self.enable_noise_reduction:
            features.append("DeepFilter")
        if self.enable_voice_analysis:
            features.append("Voice Analysis")
        
        feature_str = f" + {', '.join(features)}" if features else ""
        print(f"   🎤 Enhanced call active{feature_str}!\n")
        
        self.current_outbound_call_id = call_id
        self.call_status = CallStatus.IN_PROGRESS
        return True
    
    def send_ack(self, call_id, tag, destination, response):
        """Send ACK message"""
        to_tag = ""
        to_match = re.search(r'To:.*?tag=([^;\r\n\s]+)', response)
        if to_match:
            to_tag = f";tag={to_match.group(1)}"
        
        via = self.sip_bridge.extract_header(response, "Via")
        if not via:
            via = f"SIP/2.0/UDP {self.sip_bridge.local_ip}:{self.sip_bridge.local_port}"
        
        cseq_match = re.search(r'CSeq: (\d+)', response)
        cseq_num = cseq_match.group(1) if cseq_match else "1"
        
        ack = f"""ACK sip:{destination}@{self.sip_bridge.server}:5060 SIP/2.0
Via: {via}
Max-Forwards: 70
From: "{self.sip_bridge.extension}"<sip:{self.sip_bridge.extension}@{self.sip_bridge.server}:5060>;tag={tag}
To: <sip:{destination}@{self.sip_bridge.server}:5060>{to_tag}
Call-ID: {call_id}
CSeq: {cseq_num} ACK
Content-Length: 0

""".replace('\n', '\r\n')
        
        self.sip_bridge.send_sip_message(ack)
        print("   ➡️ Sent: ACK")
    
    def hangup_call(self):
        """Enhanced SIP hangup"""
        if not self.current_outbound_call_id:
            return
        
        if self.current_outbound_call_id in self.active_calls:
            call_info = self.active_calls[self.current_outbound_call_id]
            destination = call_info.get('destination')
            
            branch = f"z9hG4bK{uuid.uuid4().hex[:8]}"
            bye = f"""BYE sip:{destination}@{self.sip_bridge.server}:5060 SIP/2.0
Via: SIP/2.0/UDP {self.sip_bridge.local_ip}:{self.sip_bridge.local_port};branch={branch};rport
Max-Forwards: 70
From: {call_info['from']}
To: {call_info['to']}
Call-ID: {self.current_outbound_call_id}
CSeq: 3 BYE
Content-Length: 0

""".replace('\n', '\r\n')
            
            self.sip_bridge.send_sip_message(bye, call_info['addr'])
            print("   ➡️ Sent: BYE")
            
            del self.active_calls[self.current_outbound_call_id]
        
        if self.rtp_handler:
            self.rtp_handler.close()
            self.rtp_handler = None
        
        self.current_outbound_call_id = None
        print("   🔴 SIP call terminated")
    
    def keep_registration_alive(self):
        """Re-register periodically"""
        while self.sip_bridge.running:
            time.sleep(60)
            if not self.sip_bridge.running or not hasattr(self, 'active_calls'):
                break
                
            if self.sip_bridge.registered and len(self.active_calls) == 0:
                try:
                    if hasattr(self.sip_bridge, 'sock') and self.sip_bridge.sock:
                        self.sip_bridge.register()
                except OSError as e:
                    if "10038" in str(e):
                        print("   ℹ️ Registration thread exiting (socket closed)")
                        break
                except Exception as e:
                    print(f"   ⚠️ Registration refresh error: {e}")
                    break
    
    def cleanup(self):
        """Enhanced cleanup with all components"""
        print("\n🧹 Enhanced cleanup starting...")
        self.running = False
        self.hangup_event.set()
        
        # Stop enhancements
        if self.audio_enhancer:
            self.audio_enhancer.stop_processing()
            print("   🔧 DeepFilter stopped")
        
        if self.voice_analyzer:
            print("   🧠 Voice analysis stopped")
        
        # Stop SIP bridge
        if hasattr(self, 'sip_bridge'):
            self.sip_bridge.running = False
        
        if self.rtp_handler:
            self.rtp_handler.close()
        
        # Print final statistics
        if self.enable_noise_reduction and self.audio_stats["total_chunks"] > 0:
            enhancement_rate = (self.audio_stats["enhanced_chunks"] / self.audio_stats["total_chunks"]) * 100
            print(f"   📊 Final Stats: {enhancement_rate:.1f}% audio enhanced")
        
        if self.enable_voice_analysis:
            print(f"   🧠 Voice Analysis: {self.audio_stats['voice_analysis_samples']} samples processed")
        
        self.sip_bridge.cleanup()
        print("✅ Enhanced cleanup complete")