from RealtimeSTT import AudioToTextRecorder
from openai import OpenAI
import os
import json
import pygame
import tempfile
import threading
import soundfile as sf
import numpy as np
import time
import queue

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'qwen3-0.6b')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:8080')
OPENAI_TTS_API_KEY = os.getenv('OPENAI_TTS_API_KEY', OPENAI_API_KEY)
OPENAI_TTS_BASE_URL = os.getenv('OPENAI_TTS_BASE_URL', OPENAI_BASE_URL)
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', '')
OPENAI_TTS_MODEL = os.getenv('OPENAI_TTS_MODEL', 'voice-en-us-amy-low')
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'whisper-large-turbo-q5_0')
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', 'it')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful assistant.')
BACKGROUND_AUDIO = os.getenv('BACKGROUND_AUDIO', 'false').lower() in ['true', '1', 'yes', 'on']
DEBUG_PLAYBACK = os.getenv('DEBUG_PLAYBACK', 'false').lower() in ['true', '1', 'yes', 'on']
WAKE_WORD = os.getenv('WAKE_WORD', '')  # Empty string means no wake word
MCP_MODE = os.getenv('MCP_MODE', 'false').lower() in ['true', '1', 'yes', 'on']
STREAMING_TTS = os.getenv('STREAMING_TTS', 'true').lower() in ['true', '1', 'yes', 'on']
STREAM_BUFFER_SIZE = int(os.getenv('STREAM_BUFFER_SIZE', '4096'))  # Bytes to buffer before starting playback
TTS_FORMAT = os.getenv('TTS_FORMAT', 'mp3')  # Options: mp3, opus, aac, flac, wav, pcm

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Initialize OpenAI TTS client
tts_client = OpenAI(
    api_key=OPENAI_TTS_API_KEY,
    base_url=OPENAI_TTS_BASE_URL
)

# Initialize OpenAI MCP client
mcpclient = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL+"/mcp"
)


# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Conversation history
conversation_history = []

# Global variables for audio control
audio_playing = False
audio_thread = None
stop_playback_event = threading.Event()

def start_callback():
    global audio_playing
    print("🎤 Recording started!")
    
    # Stop any playing audio when recording starts (only if background audio is enabled)
    if BACKGROUND_AUDIO and audio_playing:
        print("🔇 Stopping audio for recording...")
        stop_playback_event.set()
        pygame.mixer.music.stop()
        audio_playing = False

def stop_callback():
    print("⏹️ Recording stopped!")


class StreamingAudioPlayer:
    """Handles streaming audio playback with buffering"""
    
    def __init__(self, buffer_size=4096, audio_format='mp3'):
        self.buffer_size = buffer_size
        self.audio_format = audio_format
        self.audio_queue = queue.Queue()
        self.is_streaming = True
        self.playback_thread = None
        self.temp_file = None
        self.temp_file_path = None
        self.total_bytes = 0
        self.stop_event = threading.Event()
        
    def start_streaming(self):
        """Initialize streaming session"""
        self.is_streaming = True
        self.total_bytes = 0
        self.audio_queue = queue.Queue()
        self.stop_event.clear()
        
        # Create temporary file for buffering with appropriate extension
        file_extension = f'.{self.audio_format}'
        fd, self.temp_file_path = tempfile.mkstemp(suffix=file_extension)
        self.temp_file = os.fdopen(fd, 'wb')
        
    def add_chunk(self, chunk):
        """Add audio chunk to the stream"""
        if chunk and self.is_streaming:
            self.audio_queue.put(chunk)
            self.total_bytes += len(chunk)
            
    def finish_streaming(self):
        """Signal that streaming is complete"""
        self.is_streaming = False
        self.audio_queue.put(None)  # Sentinel to signal end of stream
        
    def _write_chunks_to_file(self):
        """Write all chunks from queue to file"""
        chunks_written = 0
        
        while True:
            try:
                # Wait for chunks with timeout
                chunk = self.audio_queue.get(timeout=1.0)
                
                if chunk is None:  # End of stream
                    break
                    
                # Write chunk to file
                self.temp_file.write(chunk)
                self.temp_file.flush()  # Flush after each chunk for immediate availability
                chunks_written += len(chunk)
                
            except queue.Empty:
                # If streaming is done and queue is empty, we're finished
                if not self.is_streaming:
                    break
                continue
        
        # Ensure all data is flushed and file is closed
        self.temp_file.flush()
        self.temp_file.close()
        
        return chunks_written
        
    def play_stream_background(self):
        """Play streamed audio in background thread"""
        global audio_playing
        
        def playback_worker():
            global audio_playing
            
            try:
                audio_playing = True
                
                # Start a thread to write chunks to file
                write_thread = threading.Thread(target=self._write_chunks_to_file, daemon=True)
                write_thread.start()
                
                # Wait for minimum buffer OR streaming to complete
                bytes_buffered = 0
                buffer_ready = False
                
                print(f"📥 Buffering audio (target: {self.buffer_size} bytes)...")
                
                # Wait for either buffer size or completion
                while not buffer_ready:
                    time.sleep(0.05)  # Check more frequently
                    
                    # Check current file size
                    try:
                        bytes_buffered = os.path.getsize(self.temp_file_path)
                        # Buffer is ready if we have enough data OR streaming is done
                        if bytes_buffered >= self.buffer_size or (not self.is_streaming and bytes_buffered > 0):
                            buffer_ready = True
                    except:
                        continue
                    
                    # Timeout protection - if streaming done and no data, exit
                    if not self.is_streaming and not write_thread.is_alive() and bytes_buffered == 0:
                        print("⚠️ Streaming completed but no audio data received")
                        audio_playing = False
                        return
                
                print(f"🔊 Buffer ready ({bytes_buffered} bytes), waiting for complete file...")
                
                # Wait for write thread to finish writing all chunks
                write_thread.join(timeout=30)  # 30 second timeout for safety
                
                # Get final file size
                final_size = os.path.getsize(self.temp_file_path)
                
                if final_size == 0:
                    print("⚠️ No audio data in file")
                    audio_playing = False
                    self._cleanup()
                    return
                
                print(f"🔊 Starting playback ({final_size} bytes)")
                
                # Load and play the complete audio file
                try:
                    pygame.mixer.music.load(self.temp_file_path)
                    pygame.mixer.music.play()
                except Exception as e:
                    print(f"❌ Error loading audio: {e}")
                    audio_playing = False
                    self._cleanup()
                    return
                
                # Wait for playback to finish or stop signal
                while pygame.mixer.music.get_busy() and not self.stop_event.is_set() and not stop_playback_event.is_set():
                    pygame.time.wait(100)
                
                print(f"✅ Streaming playback completed ({self.total_bytes} bytes total)")
                
            except Exception as e:
                print(f"❌ Streaming playback error: {str(e)}")
            finally:
                audio_playing = False
                self._cleanup()
        
        # Start playback in background thread
        self.playback_thread = threading.Thread(target=playback_worker, daemon=True)
        self.playback_thread.start()
        
    def play_stream_blocking(self):
        """Play streamed audio in foreground (blocking)"""
        global audio_playing
        
        try:
            audio_playing = True
            
            # Write all chunks to file
            print(f"📥 Buffering audio...")
            chunks_written = self._write_chunks_to_file()
            
            if chunks_written == 0:
                print("⚠️ No audio data received")
                audio_playing = False
                self._cleanup()
                return
            
            print(f"🔊 Starting playback ({chunks_written} bytes)")
            
            # Load and play the audio file
            try:
                pygame.mixer.music.load(self.temp_file_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"❌ Error loading audio: {e}")
                audio_playing = False
                self._cleanup()
                return
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            print(f"✅ Playback completed ({self.total_bytes} bytes total)")
            
        except Exception as e:
            print(f"❌ Streaming playback error: {str(e)}")
        finally:
            audio_playing = False
            self._cleanup()
    
    def _cleanup(self):
        """Clean up temporary file"""
        try:
            if self.temp_file and not self.temp_file.closed:
                self.temp_file.close()
        except:
            pass
        
        try:
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")


def play_audio_background(audio_file_path):
    """Play audio in background thread"""
    global audio_playing
    
    def audio_worker():
        global audio_playing
        try:
            audio_playing = True
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy() and audio_playing and not stop_playback_event.is_set():
                pygame.time.wait(100)
            
            # Clean up
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            
            audio_playing = False
            print("✅ Background audio playback completed")
            
        except Exception as e:
            print(f"❌ Background audio error: {str(e)}")
            audio_playing = False
            # Clean up file if it exists
            if os.path.exists(audio_file_path):
                try:
                    os.unlink(audio_file_path)
                except:
                    pass
    
    # Start audio in background thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

def play_audio_blocking(audio_file_path):
    """Play audio in foreground (blocking)"""
    global audio_playing
    try:
        audio_playing = True
        pygame.mixer.music.load(audio_file_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Clean up temporary file
        os.unlink(audio_file_path)
        
        audio_playing = False
        print("✅ Speech playback completed")
        
    except Exception as e:
        print(f"❌ Blocking audio error: {str(e)}")
        audio_playing = False
        # Clean up file if it exists
        if os.path.exists(audio_file_path):
            try:
                os.unlink(audio_file_path)
            except:
                pass

def clean_text_for_tts(text):
    """Clean text for TTS by removing markdown, emojis and newlines"""
    import re
    
    # Remove emojis and special characters
    # Remove emoji ranges
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emoticons
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Misc Symbols and Pictographs
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport and Map
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # Regional indicator symbols
    text = re.sub(r'[\U00002600-\U000026FF]', '', text)  # Miscellaneous symbols
    text = re.sub(r'[\U00002700-\U000027BF]', '', text)  # Dingbats
    text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)  # Supplemental Symbols and Pictographs
    text = re.sub(r'[\U0001FA70-\U0001FAFF]', '', text)  # Symbols and Pictographs Extended-A
    
    # Remove other common emoji patterns
    text = re.sub(r'[🔥🚀💡🎤⏹️📊✅❌⚠️📝🎤🗣️🌐🔑🎯💤🤖]', '', text)
    
    # Remove markdown formatting
    # Remove bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
    text = re.sub(r'__(.*?)__', r'\1', text)      # __bold__
    text = re.sub(r'_(.*?)_', r'\1', text)        # _italic_
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # ```code```
    text = re.sub(r'`(.*?)`', r'\1', text)        # `code`
    
    # Remove headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # # Header
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url)
    
    # Remove newlines and extra whitespace
    text = re.sub(r'\n+', ' ', text)              # Replace newlines with space
    text = re.sub(r'\s+', ' ', text)             # Replace multiple spaces with single space
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def text_to_speech_streaming(text):
    """Convert text to speech using OpenAI TTS API with streaming"""
    try:
        print("📊 Generating speech (streaming)...")
        stop_playback_event.clear()
        
        # Clean text for TTS
        cleaned_text = clean_text_for_tts(text)
        
        if not cleaned_text:
            print("⚠️ No text to synthesize after cleaning")
            return
        
        # Create streaming player with configured format
        player = StreamingAudioPlayer(buffer_size=STREAM_BUFFER_SIZE, audio_format=TTS_FORMAT)
        player.start_streaming()
        
        # Start playback thread if background mode
        if BACKGROUND_AUDIO:
            player.play_stream_background()
        
        # Call OpenAI TTS API with streaming
        try:
            with tts_client.audio.speech.with_streaming_response.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=cleaned_text,
                response_format=TTS_FORMAT
            ) as response:
                # Stream audio chunks
                for chunk in response.iter_bytes(chunk_size=1024):
                    if stop_playback_event.is_set():
                        print("🔇 Streaming interrupted by user")
                        break
                    player.add_chunk(chunk)
        except Exception as e:
            print(f"❌ TTS API Error: {str(e)}")
            player.finish_streaming()
            return
        
        # Signal end of streaming
        player.finish_streaming()
        
        # If blocking mode, wait for playback to complete
        if not BACKGROUND_AUDIO:
            player.play_stream_blocking()
        
    except Exception as e:
        print(f"❌ Streaming TTS Error: {str(e)}")

def text_to_speech_legacy(text):
    """Convert text to speech using OpenAI TTS API (legacy non-streaming)"""
    try:
        print("📊 Generating speech (legacy mode)...")
        
        # Clean text for TTS
        cleaned_text = clean_text_for_tts(text)
        
        if not cleaned_text:
            print("⚠️ No text to synthesize after cleaning")
            return
        
        # Call OpenAI TTS API
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=cleaned_text
        )
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # Play audio based on configuration
        if BACKGROUND_AUDIO:
            # Play audio in background (non-blocking)
            play_audio_background(temp_file_path)
        else:
            # Play audio in foreground (blocking)
            play_audio_blocking(temp_file_path)
        
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")

def text_to_speech(text):
    """Convert text to speech - uses streaming or legacy based on configuration"""
    if STREAMING_TTS:
        text_to_speech_streaming(text)
    else:
        text_to_speech_legacy(text)

def get_openai_response(user_text):
    """Get response from OpenAI API"""
    try:
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": user_text})
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversation_history[-10:]  # Keep last 10 messages for context

        if MCP_MODE:
            print("🤖 MCP Mode: ", MCP_MODE)
            response = mcpclient.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
        else:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def validate_config():
    """Validate configuration and exit if invalid"""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("💡 Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

def process_text(text):
    """Process the recorded text and get AI response"""
    if text and text.strip():
        print(f"\n💤 You: {text}")
        
        # Get AI response
        ai_response = get_openai_response(text)
        print(f"🤖 Jarvis: {ai_response}")
        
        # Convert response to speech and play
        if ai_response:
            text_to_speech(ai_response)
        
        print()  # Add spacing after response
        return ai_response
    return None

def play_wav_file(wav_file_path):
    """Play a WAV file for debugging"""
    try:
        print(f"📊 Playing WAV file: {wav_file_path}")
        
        # Check if file exists
        if not os.path.exists(wav_file_path):
            print(f"❌ WAV file not found: {wav_file_path}")
            return False
        
        # Get file size for debugging
        file_size = os.path.getsize(wav_file_path)
        print(f"📝 File size: {file_size} bytes")
        
        # Play the audio file
        pygame.mixer.music.load(wav_file_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        print("✅ WAV file playback completed")
        return True
        
    except Exception as e:
        print(f"❌ Error playing WAV file: {str(e)}")
        return False

def save_audio_data_as_wav(audio_data, sample_rate=16000, filename="recorded.wav", play_for_debug=False):
    """Save audio data (numpy array) as a WAV file"""
    try:
        if audio_data is not None and len(audio_data) > 0:
            # Ensure audio data is in the correct format (float32, normalized)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Save as WAV file
            sf.write(filename, audio_data, sample_rate)
            print(f"💾 Audio saved as {filename} (shape: {audio_data.shape})")
            
            # Play for debugging if requested
            if play_for_debug:
                play_wav_file(filename)
            
            return filename
        else:
            print("❌ No audio data to save")
            return None
    except Exception as e:
        print(f"❌ Error saving audio data: {str(e)}")
        return None

def save_audio_frames_as_wav(frames, sample_rate=16000, filename="recorded.wav"):
    """Save audio frames as a WAV file"""
    try:
        # Convert frames to numpy array
        if frames:
            # Join all frames into a single byte string
            audio_bytes = b''.join(frames)
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Save as WAV file
            sf.write(filename, audio_float, sample_rate)
            print(f"💾 Audio saved as {filename}")
            return filename
        else:
            print("❌ No audio frames to save")
            return None
    except Exception as e:
        print(f"❌ Error saving audio: {str(e)}")
        return None

def transcribe_wav_with_openai(wav_file_path):
    """Transcribe WAV file using OpenAI Whisper API"""
    try:
        print("🎤 Transcribing with OpenAI Whisper...")
        with open(wav_file_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language=WHISPER_LANGUAGE  # Italian language
            )
        
        return transcript.text
    except Exception as e:
        print(f"❌ OpenAI Whisper transcription error: {str(e)}")
        return None

class CustomAudioRecorder(AudioToTextRecorder):
    """Custom AudioToTextRecorder that allows access to audio data"""
    
    def get_audio_data(self):
        """Get the processed audio data as numpy array"""
        return self.audio if hasattr(self, 'audio') and self.audio is not None else None
    
    def get_audio_frames(self):
        """Get the current audio frames (before processing)"""
        return self.frames if hasattr(self, 'frames') else []
    
    def get_last_audio_frames(self):
        """Get the last recorded audio frames (before processing)"""
        return self.last_frames if hasattr(self, 'last_frames') else []
    
    def clear_frames(self):
        """Clear the audio frames and data"""
        if hasattr(self, 'frames'):
            self.frames.clear()
        if hasattr(self, 'last_frames'):
            self.last_frames.clear()
        if hasattr(self, 'audio'):
            self.audio = None

def process_audio_with_openai_whisper(recorder):
    """Process recorded audio using OpenAI Whisper API"""
    try:
        print("📊 Processing audio with OpenAI Whisper...")
        
        # Get the processed audio data (numpy array)
        audio_data = recorder.get_audio_data()
        print(f"📊 Audio data available: {audio_data is not None}")
        
        if audio_data is not None:
            print(f"📊 Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Save audio data as WAV file
            wav_filename = f"temp_audio_{int(time.time())}.wav"
            wav_path = save_audio_data_as_wav(audio_data, sample_rate=16000, filename=wav_filename, play_for_debug=DEBUG_PLAYBACK)
            
            if wav_path:
                # Transcribe using OpenAI Whisper
                transcription = transcribe_wav_with_openai(wav_path)
                
                # Clean up temporary file
                try:
                    os.unlink(wav_path)
                except:
                    pass
                
                return transcription
            else:
                return None
        else:
            print("❌ No audio data available")
            return None
            
    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        return None

if __name__ == '__main__':
    # Validate configuration
    validate_config()
    
    print("🚀 Jarvis Voice Assistant Started!")
    print("💡 Configuration:")
    print(f"   📝 Chat Model: {OPENAI_MODEL}")
    print(f"   🎤 TTS Model: {OPENAI_TTS_MODEL}")
    print(f"   🗣️  TTS Voice: {OPENAI_TTS_VOICE}")
    print(f"   🌐 Base URL: {OPENAI_BASE_URL}")
    print(f"   📊 Background Audio: {'✅ Enabled' if BACKGROUND_AUDIO else '❌ Disabled'}")
    print(f"   🎵 Debug Playback: {'✅ Enabled' if DEBUG_PLAYBACK else '❌ Disabled'}")
    print(f"   🎯 Wake Word: {'✅ ' + WAKE_WORD if WAKE_WORD else '❌ Disabled'}")
    print(f"   🎼 Streaming TTS: {'✅ Enabled' if STREAMING_TTS else '❌ Disabled'}")
    if STREAMING_TTS:
        print(f"   📦 Stream Buffer: {STREAM_BUFFER_SIZE} bytes")
        print(f"   🎵 Audio Format: {TTS_FORMAT}")
    print(f"   🔑 API Key: ✅ Set")
    print("🎯 Starting voice assistant...")
    if WAKE_WORD:
        print(f"💡 Say '{WAKE_WORD}' to activate the assistant!\n")
    else:
        print("💡 Just start speaking - no wake word needed!\n")
    
    # Configure wake word based on environment variable
    wake_word_config = {}
    if WAKE_WORD:
        wake_word_config["wake_words"] = WAKE_WORD
        wake_word_config["wakeword_backend"] = "pvporcupine"  # Enable wake word backend
    else:
        wake_word_config["wakeword_backend"] = ""  # Disable wake word backend
    
    recorder = CustomAudioRecorder(on_recording_start=start_callback,
                                 model="tiny",  # Use smallest model since we're not using it for transcription
                                 on_recording_stop=stop_callback,
                                 **wake_word_config)
    
    # Start listening for voice activity
    recorder.listen()
    
    while True:
        try:
            print("🎧 Waiting for voice input...")
            
            # Use the built-in wait_audio method to handle voice activity detection
            recorder.wait_audio()
            
            print("✅ Audio recording completed, processing...")
            # Process the recorded audio with OpenAI Whisper
            recorded_text = process_audio_with_openai_whisper(recorder)
            
            if recorded_text:
                # Process the transcribed text
                process_text(recorded_text)
            else:
                print("⚠️ No audio was captured, please try again")
            
            # Clear frames for next recording
            recorder.clear_frames()
            
            # Start listening again for the next interaction
            recorder.listen()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {str(e)}")
            continue
