import torch
import numpy as np
import soundfile as sf
import os
import io
import gc

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

