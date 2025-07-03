from scipy.io.wavfile import write
from orpheus_cpp import OrpheusCpp
import numpy as np
from typing import Literal
from huggingface_hub import hf_hub_download
import onnxruntime



class OrpheusCppCuda(OrpheusCpp):
    def __init__(
        self,
        n_gpu_layers: int = 0,
        n_threads: int = 0,
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
            n_ctx=4096,
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

orpheus = OrpheusCppCuda(verbose=False, lang="es",n_gpu_layers=-1,n_threads=2)

text = "[javi] Te voy contar mi historia... Eu Nací y crecí, sabiendo que era diferente. Me atraía la belleza y la expresión. Mi familia, aunque al principio le costó, me aceptó. Con el tiempo, me convertí en Ernesto, el travesti familiar. Es un título que llevo con orgullo; soy el que alegra las fiestas y el que siempre les recuerda a todos que ser uno mismo es la verdadera libertad."
buffer = []
for i, (sr, chunk) in enumerate(orpheus.stream_tts_sync(text, options={"voice_id": "javi"})):
   buffer.append(chunk)
   print(f"Generated chunk {i}")
buffer = np.concatenate(buffer, axis=1)
write("output.wav", 24_000, np.concatenate(buffer))
