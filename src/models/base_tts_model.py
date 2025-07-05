import torch
import numpy as np
import soundfile as sf
import os
import io
import gc
import sounddevice as sd
import soundfile as sf

from abc import ABC, abstractmethod

from src.utils import log_status, Color

class BaseTTSModel(ABC):
    """
    Abstract Base Class for Text-to-Speech models.
    Defines the common interface and shared initialization logic for TTS models,
    ensuring consistency across different implementations.
    """
    def __init__(
        self,
        model_name: str,
        device: str = None,
        generation_config_defaults: dict = None,
    ):
        if model_name: self.model_name = model_name
        
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_status(f"BaseTTSModel will use device: {self.device}", Color.BLUE)
        
        self.sampling_rate = 24_000
        self.generation_config_defaults = {
            'do_sample': True,
            'max_new_tokens': 4096,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 0.7,
        }
        if generation_config_defaults:
            self.generation_config_defaults.update(generation_config_defaults)
    
    def nomalize_to_wave_bytes(self, audio, sampling_rate=24_000):
        audio_int16 = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_int16, samplerate=sampling_rate, format='WAV')
        wav_bytes = wav_buffer.getvalue()
        wav_buffer.close()
        return wav_bytes

    @abstractmethod
    def _load_model_and_processor(self):
        """
        Abstract method to load the specific TTS model and its associated processor.
        Concrete subclasses must implement this.
        This method should handle model loading, device placement, and error handling specific to the model.
        """
        pass
    
    def save_audio_bytes(self, audio_bytes: bytes, output_filename: str):
        """
        Save WAV bytes to a file, creating directories if needed.
        """
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_status(f"Created output directory: {output_dir}", Color.BLUE)

        with open(output_filename, "wb") as f:
            f.write(audio_bytes)
        log_status(f"Audio successfully saved to: {output_filename}", Color.GREEN)

    def play_audio_bytes(self, audio_bytes: bytes):
        """
        Play WAV audio bytes directly using sounddevice.
        """
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_buffer, dtype='float32')
            log_status(f"Playing audio ({len(audio_bytes)} bytes, {samplerate} Hz)...", Color.BLUE)
            sd.play(data, samplerate)
            sd.wait()  # Wait until playback finishes
            log_status("Playback finished.", Color.GREEN)
        except Exception as e:
            log_status(f"Error during audio playback: {e}", Color.RED)

    def play_audio_chunk(chunk: np.ndarray, samplerate: int):
        """
        Play a single audio chunk (NumPy array) using sounddevice.

        Args:
            chunk (np.ndarray): The audio data, shape (1, N), (N,), or (N, 1).
            samplerate (int): The sample rate of the audio chunk.
        """
        if chunk.ndim == 2 and chunk.shape[0] == 1:
            # Transpose from (1, N) to (N, 1)
            chunk = chunk.T
        elif chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]  # Shape (N,) → (N, 1)

        try:
            sd.play(chunk.astype(np.float32), samplerate)
            sd.wait()  # Wait until chunk finishes playing
        except Exception as e:
            print(f"Playback error: {e}")

    @abstractmethod
    def generate_audio_bytes(
        self,
        text: str,
        language: str = "en", 
        generation_params: dict = None,
        output_filename="output.wav",
        **kargs
    ) -> bytes:
        """
        Abstract method to generate audio bytes from text.
        Concrete subclasses must implement this.
        Returns audio as WAV-formatted bytes.
        """
        pass

    def __del__(self):
        """Base destructor for common resource cleanup (e.g., PyTorch models)."""
        # Note: Specific model/processor deletion should be handled in concrete classes
        # to ensure proper cleanup of their specific objects.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() # Explicitly call garbage collection
        log_status(f"BaseTTSModel resources for {self.model_name} cleaned up.", Color.BLUE)

