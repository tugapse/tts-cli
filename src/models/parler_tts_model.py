# tts_app/src/parler_tts_model.py

import torch
import numpy as np
import soundfile as sf
import os
import io
import gc

from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
import requests.exceptions

from src.models.base_tts_model import BaseTTSModel
from src.utils import log_status, Color

class ParlerTTSModel(BaseTTSModel):
    """
    Concrete implementation of BaseTTSModel for Hugging Face's Parler-TTS models.
    Closely matches the official example for parler-tts-mini-v1 for reliable operation.
    """
    def __init__(
        self,
        model_name: str = None, # Changed default model name
        device: str = None,
        generation_config_defaults: dict = None,
    ):
        self.main_tokenizer = None
        self.description_tokenizer = None
        self.model = None 
        super().__init__(model_name, device, generation_config_defaults)
        self.model_name = model_name or "parler-tts/parler-tts-mini-v1"

        try:
            self._load_model_and_processor()
        except Exception as e:
            log_status(f"ERROR: ParlerTTSModel initialization failed during model loading: {e}", Color.RED)
            raise

    def _load_model_and_processor(self):
        """
        Loads the Parler-TTS tokenizers and model from Hugging Face.
        Uses AutoTokenizer for both text and description, matching the official example.
        """
        log_status(f"Attempting to load Parler-TTS model: {self.model_name}...", Color.YELLOW)
        try:
            self.main_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            load_kwargs = {}
            if self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16

            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            self.model.to(self.device) 
            self.sampling_rate = self.model.config.sampling_rate

            self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)

            log_status(f"Successfully loaded Parler-TTS model: {self.model_name}", Color.GREEN)

        except GatedRepoError as e:
            log_status(f"ERROR: Failed to load gated model '{self.model_name}'. Access denied or not authenticated. Details: {e}", Color.RED)
            raise
        except RepositoryNotFoundError:
            log_status(f"ERROR: Model '{self.model_name}' not found on Hugging Face Hub. Check spelling.", Color.RED)
            raise
        except requests.exceptions.HTTPError as e:
            log_status(f"ERROR: Could not download model files for '{self.model_name}'. Check network, disk space, or proxy settings. Details: {e}", Color.RED)
            raise
        except Exception as e:
            log_status(f"CRITICAL ERROR: Model loading failed for {self.model_name}: {e}", Color.RED)
            raise

    def generate_audio_bytes(
        self,
        text: str,
        language: str = "en",  # Kept for compatibility, not used here
        speaker_embedding: torch.Tensor = None,  # Not used here
        generation_params: dict = None,
        description_prompt: str = "A clear voice with a neutral tone.",
        output_filename="output.wav"
    ) -> bytes:
        if self.model is None or self.main_tokenizer is None or self.description_tokenizer is None:
            raise RuntimeError("ParlerTTS model or tokenizers not loaded.")

        if not text.strip():
            log_status("WARNING: Attempted to synthesize speech from empty text. Returning empty bytes.", Color.YELLOW)
            return b""

        log_status(f"Generating audio for text: '{text[:50]}{'...' if len(text) > 50 else ''}'", Color.YELLOW)
        log_status(f"Using description prompt: '{description_prompt}'", Color.CYAN)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Prepare inputs using the main tokenizer for the text prompt
        prompt_inputs = self.main_tokenizer(
            text, 
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        # Prepare inputs using the description tokenizer for the voice description
        description_inputs = self.description_tokenizer(
            description_prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        final_generation_params = self.generation_config_defaults.copy()
        if generation_params:
            final_generation_params.update(generation_params)

        with torch.no_grad():
            audio_tensor = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                **final_generation_params
            ).cpu().numpy()

        audio = audio_tensor.squeeze()
        audio_bytes = self.nomalize_to_wave_bytes(audio=audio, sampling_rate=self.model.config.sampling_rate)
        log_status(f"Audio generation complete. Generated {len(audio_bytes)} bytes.", Color.GREEN)

        return audio_bytes


    def __del__(self):
        """Clean up ParlerTTS specific resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'main_tokenizer') and self.main_tokenizer is not None:
            del self.main_tokenizer
        if hasattr(self, 'description_tokenizer') and self.description_tokenizer is not None:
            del self.description_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 
        super().__del__()

