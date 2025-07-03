import torch
import numpy as np
import soundfile as sf
import os
import io
import gc
from typing import Literal

from scipy.io.wavfile import write
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
import requests.exceptions
from orpheus_cpp import OrpheusCpp
import onnxruntime

from src.models.base_tts_model import BaseTTSModel
from src.utils import log_status, Color





class OrpheusCppCuda(OrpheusCpp):
    def __init__(
        self,
        n_gpu_layers: int = 0,
        n_threads: int = 0,
        n_context:int = 4096,
        verbose: bool = True,
        lang: Literal["en", "es", "ko", "fr"] = "es",
    ):
        import importlib.util

        if importlib.util.find_spec("llama_cpp") is None:
            if sys.platform == "darwin":
                # Check if macOS 11.0+ on arm64 (Apple Silicon)
                is_arm64 = platform.machine() == "arm64"
                version = platform.mac_ver()[0].split(".")
                is_macos_11_plus = len(version) >= 2 and int(version[0]) >= 11
                is_macos_10_less = len(version) >= 2 and int(version[0]) < 11

                if is_arm64 and is_macos_11_plus:
                    extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal"
                elif is_macos_10_less:
                    raise ImportError(
                        "llama_cpp does not have pre-built wheels for macOS 10.x "
                        "Follow install instructions at https://github.com/abetlen/llama-cpp-python"
                    )
                else:
                    extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            else:
                extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"

            raise ImportError(
                f"llama_cpp is not installed. Please install it using `pip install llama-cpp-python {extra_index_url}`."
            )
        repo_id = self.lang_to_model[lang]
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=repo_id.split("/")[-1].lower().replace("-gguf", ".gguf"),
        )
        from llama_cpp import Llama

        if n_gpu_layers == 0:
            print(
                "Running model without GPU Acceleration. Please set n_gpu_layers parameters to control the number of layers to offload to GPU."
            )

        self._llm = Llama(
            model_path=model_file,
            n_ctx=n_context,
            verbose=verbose,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            batch_size=1,
        )

        repo_id = "onnx-community/snac_24khz-ONNX"
        snac_model_file = "decoder_model.onnx"
        snac_model_path = hf_hub_download(
            repo_id, subfolder="onnx", filename=snac_model_file
        )

        # Load SNAC model with optimizations
        self._snac_session = onnxruntime.InferenceSession(
            snac_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    


class OrpheusTTSModel(BaseTTSModel):
    """
    Concrete implementation of BaseTTSModel for Hugging Face's Parler-TTS models.
    Closely matches the official example for parler-tts-mini-v1 for reliable operation.
    """
    def __init__(
        self,
        model_name: str = None, 
        device: str = None,
        generation_config_defaults: dict = None,
    ):
        self.main_tokenizer = None
        self.description_tokenizer = None
        self.model = None 
        super().__init__(model_name, device, generation_config_defaults)
        self.model_name = model_name or "canopylabs/orpheus-tts-0.1-finetune-prod"

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
        log_status(f"Attempting to load Orpheus-TTS model: {self.model_name}...", Color.YELLOW)

        n_gpu_layers = self.generation_config_defaults.get("n_gpu_layers",-1)
        n_threads = self.generation_config_defaults.get("n_threads",1)
        n_ctx = self.generation_config_defaults.get("n_ctx",4096)
        lang = self.generation_config_defaults.get("lang","en")
        log_status(f"Loading params: n_gpu_layers:{n_gpu_layers}, n_threads:{n_threads}, n_ctx:{n_ctx},lang:{lang}", Color.HEADER)


        self.model = OrpheusCppCuda(
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_context=n_ctx,
            verbose=False,
            lang=lang
            )
        log_status(f"Successfully loaded Parler-TTS model: {self.model_name}", Color.GREEN)

    def generate_audio_bytes(
        self,
        text: str,
        language: str = "en", # Kept for BaseTTSModel compatibility, but not used by ParlerTTS here
        generation_params: dict = None,
        output_filename="output.wav",
        **kargs
    ) -> bytes:
        """
        Generates audio bytes from the given text using the Parler-TTS model.
        This method is implemented from the BaseTTSModel abstract class.
        Returns audio as WAV-formatted bytes.
        
        Args:
            text (str): The text to convert to speech.
            language (str, optional): Kept for BaseTTSModel compatibility, not directly used by ParlerTTS.
            speaker_embedding (torch.Tensor, optional): Kept for BaseTTSModel compatibility, not directly used by ParlerTTS.
            generation_params (dict, optional): Dictionary of additional generation parameters
                                                to override defaults (e.g., {'temperature': 0.8}).
            description_prompt (str, optional): A text description of the desired voice characteristics
                                                (e.g., "A female speaker with a calm voice.").
        """
        if self.model is None:
            raise RuntimeError("Orpheus model not loaded.")

        if not text.strip():
            log_status("WARNING: Attempted to synthesize speech from empty text. Returning empty bytes.", Color.YELLOW)
            return b""

        log_status(f"Generating audio for text: '{text[:50]}{'...' if len(text) > 50 else ''}'", Color.YELLOW)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        buffer = []
        for i, (sr, chunk) in enumerate(self.model.stream_tts_sync(text, options={"voice_id": "Dan"})):
            buffer.append(chunk)
            print(f"Generated chunk {i}")
        buffer = np.concatenate(buffer, axis=1)

        output_dir = os.path.dirname(output_filename)
       
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_status(f"Created output directory: {output_dir}", Color.BLUE)
    
        write(output_filename,self.sampling_rate , np.concatenate(buffer))
        log_status(f"Audio successfully saved to: {output_filename}", Color.GREEN)

    def __del__(self):
        """Clean up specific resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 
        super().__del__()

