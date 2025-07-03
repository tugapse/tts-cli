import argparse
import sys
import os
import numpy as np
import torch

from src.models.model_manager import ModelManager
from src.utils import log_status, Color

__version__ = "0.2.0"
__available_model_types = ["parler", "orpheus"]

def main():
    parser = argparse.ArgumentParser(
        description=f"{Color.BOLD}{Color.BLUE}TTS CLI App:{Color.RESET} Generate speech from text using Hugging Face TTS models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- New Argument for Building/Installing Dependencies ---
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run the dependency installation script (dependency_installer.py) before running the main app."
    )
    
    # --- Argument to auto-accept prompts for both main and build scripts ---
    parser.add_argument(
        "--auto-accept",
        "-y",
        action="store_true",
        help="Automatically accept all installation prompts if --build is used."
    )

    # --- New Argument for Model Type Selection ---
    parser.add_argument(
        "--model-type",
        type=str,
        default="parler",
        choices=__available_model_types, # Restrict choices to currently supported models
        help=f"The type of TTS model to use. Currently supports {__available_model_types}."
    )

    # --- Required Argument ---
    # Make --text not required initially, as it might be skipped if only --build is used
    parser.add_argument(
        "--text",
        type=str,
        required=False, 
        help="The text string to convert to speech."
    )
    
    # --- Optional Arguments ---
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.wav",
        help="The path to save the generated WAV audio file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The Hugging Face model identifier."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="The language of the input text (e.g., 'en', 'fr', 'es'). "
             "Note: ParlerTTS primarily uses description prompt for voice, "
             "and language is inferred from text. This argument is kept for compatibility."
    )
    parser.add_argument(
        "--speaker_embedding_path",
        type=str,
        help="Optional path to a .npy file containing a speaker embedding. "
             "Note: ParlerTTS mini-v1 primarily uses description prompt for voice. "
             "This argument is kept for compatibility with other potential models."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0'). "
             "Defaults to 'cuda' if available, otherwise 'cpu'."
    )
    
    # --- Generation Parameters ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for text-to-speech generation. Higher values increase randomness."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter. Considers only the top K most likely next tokens."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter. Considers tokens whose cumulative probability "
             "exceeds P. Use in conjunction with temperature."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate. Impacts the maximum length of the audio."
    )
    parser.add_argument(
        "--description_prompt",
        type=str,
        default="A clear voice with a neutral tone.",
        help="A text description of the desired voice characteristics (e.g., 'A female speaker with a calm voice.')."
    )

    parser.add_argument(
        "--debug-console","-dc",
        action='store_true',
        help="raise erros in the console!"
    )

    args = parser.parse_args()

    # --- Handle the --build argument ---
    if args.build:
        log_status(f"{Color.BOLD}{Color.GREEN}Running dependency installer...{Color.RESET}", Color.GREEN)
        try:
            # Import the dependency_installer module
            import dependency_installer
            # Call its main function, passing the auto_accept status
            dependency_installer.main(auto_accept=args.auto_accept)
            log_status(f"{Color.BOLD}{Color.GREEN}Dependency installation completed.{Color.RESET}", Color.GREEN)
        except ImportError:
            log_status(f"{Color.RED}Error: 'dependency_installer.py' not found in the same directory.{Color.RESET}", Color.RED)
            sys.exit(1)
        except Exception as e:
            log_status(f"{Color.RED}An error occurred during dependency installation: {e}{Color.RESET}", Color.RED)
            sys.exit(1)
        
        # If only building, exit after installation
        if not args.text: # If --text is not provided, assume user only wanted to build
            log_status(f"{Color.BOLD}{Color.BLUE}Exiting after dependency installation. To run the app, provide --text.{Color.RESET}", Color.BLUE)
            sys.exit(0)

    # Ensure --text is provided if not just building
    if not args.text:
        parser.error("--text is required unless --build is specified to only install dependencies.")

    log_status(f"{Color.BOLD}--- ParlerTTS CLI App Started ---{Color.RESET}", Color.BLUE)
    log_status(f"Input Text: '{args.text[:100]}{'...' if len(args.text) > 100 else ''}'", Color.CYAN)
    log_status(f"Output Path: {args.output_path}", Color.CYAN)
    log_status(f"Model Name: {args.model_name}", Color.CYAN)
    log_status(f"Model Type: {args.model_type}", Color.CYAN) # Log the new argument
    log_status(f"Language: {args.language}", Color.CYAN)
    log_status(f"Speaker Embedding Path: {args.speaker_embedding_path if args.speaker_embedding_path else 'None (random speaker)'}", Color.CYAN)
    log_status(f"Description Prompt: '{args.description_prompt}'", Color.CYAN)
    log_status(f"Generation Params: Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, Max Tokens={args.max_new_tokens}", Color.CYAN)

    speaker_embedding_tensor = None
    if args.speaker_embedding_path:
        try:
            if not os.path.exists(args.speaker_embedding_path):
                raise FileNotFoundError(f"Speaker embedding file not found at '{args.speaker_embedding_path}'")
            speaker_embedding_tensor = torch.tensor(np.load(args.speaker_embedding_path))
            log_status(f"Loaded speaker embedding from: {args.speaker_embedding_path}", Color.GREEN)
        except Exception as e:
            log_status(f"ERROR: Failed to load speaker embedding: {e}", Color.RED)
            sys.exit(1)

    tts_engine = None
    try:
        generation_params = {
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'max_new_tokens': args.max_new_tokens,
        }

        # Use ModelManager to load the TTS engine
        model_manager = ModelManager(model_type=args.model_type)
        # Pass ParlerTTSModel class to ModelManager to avoid circular imports
        tts_engine = model_manager.load_model(
            model_name=args.model_name,
            device=args.device,
            generation_config_defaults=generation_params
        )

        tts_engine.generate_audio_bytes(
            text=args.text,
            language=args.language,
            speaker_embedding=speaker_embedding_tensor,
            generation_params=generation_params,
            description_prompt=args.description_prompt,
            output_filename=args.output_path
        )


    except Exception as e:
        log_status(f"{Color.RED}An error occurred during execution: {e}{Color.RESET}", Color.RED)
        if args.debug_console:
            raise e
        sys.exit(1)
    finally:
        if tts_engine:
            del tts_engine

if __name__ == "__main__":
    main()
