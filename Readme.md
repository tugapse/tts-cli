# ParlerTTS CLI Application

This repository contains a command-line interface (CLI) application for generating high-quality, natural-sounding speech using the models from Hugging Face.
- [Orpheus 3B 0.1 Finetuned](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
- [Parler-TTS Mini v1.1](https://huggingface.co/parler-tts/parler-tts-mini-v1.1)

The project is structured for modularity and extensibility, allowing for easy integration of different TTS backends in the future.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [System Dependencies (Arch Linux)](#system-dependencies-arch-linux)
  - [Python Environment Setup](#python-environment-setup)
  - [Python Package Installation](#python-package-installation)
  - [GPU (CUDA) Support](#gpu-cuda-support)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Controlling Voice Characteristics](#controlling-voice-characteristics)
  - [Full CLI Options](#full-cli-options)
- [Speaker Consistency Across Generations](#speaker-consistency-across-generations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Text-to-Speech Generation**: Convert text into high-fidelity audio using Huggingface models.
- **Modular Design**: Clean separation of concerns with an abstract base class for easy integration of other TTS models.
- **CLI Interface**: Simple command-line arguments for configuration and execution.
- **GPU Acceleration**: Supports CUDA for faster generation if a compatible GPU is available.

## Installation
### System Dependencies (Arch Linux)
Before installing Python packages, ensure you have the necessary system-level build tools, especially for packages like `sentencepiece` which require compilation.

```bash
sudo pacman -Syu             # Update your system first
sudo pacman -S base-devel cmake pkgconf
```

- **base-devel**: Provides essential build tools like `gcc`, `make`, etc.
- **cmake**: Required for compiling `sentencepiece`.
- **pkgconf**: Provides `pkg-config`.

### Python Environment Setup
It is highly recommended to use a stable Python version (e.g., Python 3.11 or 3.12) and a virtual environment to avoid conflicts with your system's Python packages. Python 3.13 is still in early development and may have compatibility issues with some libraries.

Navigate to your project's root directory:
```bash
cd /path/to/your/tts_app/
```

Deactivate any existing virtual environment:
```bash
deactivate
```

Delete any old virtual environment (for a clean slate):
```bash
rm -rf .venv
```

Install your preferred stable Python version (if not already installed):
```bash
sudo pacman -S python311 # Or python312 if you prefer
```

Create a new virtual environment using the stable Python version:
```bash
python3.11 -m venv .venv
```

(Replace `python3.11` with your chosen version, e.g., `python3.12`).

Activate the new virtual environment:
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

Verify the Python version in the new environment:
```bash
python --version
```

Ensure it shows Python 3.11.x (or your chosen stable version).

### Python Package Installation
Install the required Python libraries using pip and the `requirements.txt` file.
```bash
python dependency_installer.py
```

### GPU (CUDA) Support
If you have a CUDA-compatible NVIDIA GPU and want to leverage it for faster inference, you must install PyTorch specifically for your CUDA version. The `torch` entry in `requirements.txt` installs the CPU-only version by default.

Identify your CUDA version. Visit the official PyTorch website's "Get Started" section: https://pytorch.org/get-started/locally/

Select your exact configuration (e.g., OS, Package, Language, CUDA version) to get the precise installation command. Run that command after you've run `pip install -r requirements.txt`.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Example for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage
Navigate to the root directory of your `tts_app` project and activate your virtual environment before running the `main.py` script.
```bash
cd /path/to/your/tts_app/
source .venv/bin/activate
```

### Basic Usage
The `--text` argument is required.
```bash
python main.py --text "Hello, this is a test of the Parler TTS CLI application."
```

This will generate `output.wav` in your current directory.

### Full CLI Options
You can see all available arguments and their default values by running:
```bash
python main.py --help
```

```
usage: main.py [-h] --text TEXT [--output_path OUTPUT_PATH] [--model_name MODEL_NAME] [--language LANGUAGE] [--speaker_embedding_path SPEAKER_EMBEDDING_PATH] [--device DEVICE] [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P] [--max_new_tokens MAX_NEW_TOKENS] [--description_prompt DESCRIPTION_PROMPT]
ParlerTTS CLI App: Generate speech from text using Hugging Face Parler-TTS models.
options:
  -h, --help            show this help message and exit
  --text TEXT           The text string to convert to speech. (default: None)
  --output_path OUTPUT_PATH
                        The path to save the generated WAV audio file. (default: output.wav)
  --model_name MODEL_NAME
                        The Hugging Face model identifier for Parler-TTS. (default: parler-tts/parler-tts-mini-multilingual-v1.1)
  --language LANGUAGE   The language of the input text (e.g., 'en', 'fr', 'es'). Note: ParlerTTS primarily uses description prompt for voice, and language is inferred from text. This argument is kept for compatibility. (default: en)
  --speaker_embedding_path SPEAKER_EMBEDDING_PATH
                        Optional path to a .npy file containing a speaker embedding. Note: ParlerTTS mini-v1 primarily uses description prompt for voice. This argument is kept for compatibility with other potential models. (default: None)
  --device DEVICE       The device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to 'cuda' if available, otherwise 'cpu'. (default: None)
  --temperature TEMPERATURE
                        Sampling temperature for text-to-speech generation. Higher values increase randomness. (default: 0.7)
  --top_k TOP_K         Top-k sampling parameter. Considers only the top K most likely next tokens. (default: 50)
  --top_p TOP_P         Top-p (nucleus) sampling parameter. Considers tokens whose cumulative probability exceeds P. Use in conjunction with temperature. (default: 0.95)
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate. Impacts the maximum length of the audio. (default: 4096)
  --description_prompt DESCRIPTION_PROMPT
                        A text description of the desired voice characteristics (e.g., 'A female speaker with a calm voice.'). (default: A clear voice with a neutral tone.)
```

## Speaker Consistency Across Generations
ParlerTTS achieves speaker consistency primarily through the `description_prompt`. To maintain the same voice and tone across multiple generated audio segments, always use the **exact same string** for the `--description_prompt` argument for each segment.

The model was trained to interpret these natural language descriptions to synthesize speech with the specified characteristics. Experiment with descriptive phrases like:

- `"A deep, resonant male voice, speaking slowly."`
- `"An energetic female voice with a slightly high pitch."`
- `"Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."` (Example from ParlerTTS documentation for a named speaker).

## Troubleshooting
- **`ImportError: cannot import name 'AutoModelForTextToSpeech'` or "architecture not recognized"**:
    This indicates a version mismatch or caching issue.
    1. Ensure you are using a **stable Python version** (3.11 or 3.12 recommended).
    2. Perform a **clean reinstallation** of dependencies:
        ```bash
        deactivate # If active
        rm -rf .venv
        pip cache purge
        rm -rf ~/.cache/huggingface/hub/ # CAUTION: Clears all downloaded HF models
        python3.11 -m venv .venv # Or your chosen stable Python version
        source .venv/bin/activate
        pip install -r requirements.txt
        # Reinstall CUDA torch if needed
        ```
- **`subprocess.CalledProcessError: Command '['./build_bundled.sh', '0.2.0']' returned non-zero exit status 1.` (during `sentencepiece` installation)**:
    This is a compilation error.
    1. Ensure you have **system-level build tools** installed (e.g., `base-devel`, `cmake`, `pkgconf` on Arch Linux).
    2. Verify your `cmake` version is 3.5 or newer (`cmake --version`).
    3. Confirm you are using a **stable Python version** (3.11 or 3.12).
    4. Perform a **clean reinstallation** as described above.
- **"Flash attention 2 is not installed"**:
    This is a performance warning, not an error. The application will still function. For performance gains on NVIDIA GPUs, you can try:
    ```bash
    pip install flash-attn --no-build-isolation
    ```
- **`FutureWarning: torch.nn.utils.weight_norm is deprecated...`**:
    This is a deprecation warning from PyTorch. It's harmless for current functionality and does not require action from your side; it needs to be addressed by the `parler_tts` library maintainers in future updates.

## Contributing
Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

## License
This project is open-source. Please refer to the specific licenses of the underlying libraries (ParlerTTS, Transformers, PyTorch, etc.) for their respective terms of use.