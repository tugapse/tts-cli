import argparse
import sys
import os
import numpy as np
import torch

from src.models.model_manager import ModelManager
from src.utils import log_status, Color

__version__ = "0.2.2"
__available_model_types = ["parler", "orpheus"]

def main():
    parser = argparse.ArgumentParser(
        description=f"{Color.BOLD}{Color.BLUE}TTS CLI App:{Color.RESET} Generate speech from text using Hugging Face TTS models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Build/install dependencies ---
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run the dependency installation script (dependency_installer.py) before running the main app."
    )

    # --- Auto-accept prompts ---
    parser.add_argument(
        "--auto-accept",
        "-y",
        action="store_true",
        help="Automatically accept all installation prompts if --build is used."
    )

    # --- Model type ---
    parser.add_argument(
        "--model-type",
        type=str,
        default="parler",
        choices=__available_model_types,
        help=f"The type of TTS model to use. Currently supports {__available_model_types}."
    )

    # --- Text (can also come from positional or stdin) ---
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="The text string to convert to speech."
    )

    parser.add_argument(
        "input_text",
        nargs="?",
        help="Positional argument for the text to convert to speech. Used if --text is not provided."
    )

    # --- Other optional arguments ---
    parser.add_argument(
        "--output-file","-o",
        type=str,
        help="The path to save the generated WAV audio file."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The Hugging Face model identifier."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="The language of the input text."
    )
    parser.add_argument(
        "--speaker-embedding_path",
        type=str,
        help="Optional path to a .npy file containing a speaker embedding."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0')."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--description-prompt", "-dp",
        type=str,
        default="A clear voice with a neutral tone.",
        help="A text description of the desired voice characteristics."
    )
    parser.add_argument(
        "--debug-console", "-dc",
        action='store_true',
        help="Raise errors in the console for debugging."
    )
    parser.add_argument(
        "--quiet", "-q",
        action='store_true',
        help="Do not play the generated audio this is useful when you what to only save as a file."
    )

    args = parser.parse_args()

    # --- Combine possible input sources for the text ---
    piped_text = None
    if not sys.stdin.isatty():
        piped_text = sys.stdin.read().strip()

    final_text = args.text or args.input_text or piped_text

    # --- Handle --build ---
    if args.build:
        log_status(f"{Color.BOLD}{Color.GREEN}Running dependency installer...{Color.RESET}", Color.GREEN)
        try:
            import dependency_installer
            dependency_installer.main(auto_accept=args.auto_accept)
            log_status(f"{Color.BOLD}{Color.GREEN}Dependency installation completed.{Color.RESET}", Color.GREEN)
        except ImportError:
            log_status(f"{Color.RED}Error: 'dependency_installer.py' not found in the same directory.{Color.RESET}", Color.RED)
            sys.exit(1)
        except Exception as e:
            log_status(f"{Color.RED}An error occurred during dependency installation: {e}{Color.RESET}", Color.RED)
            sys.exit(1)

        if not final_text:
            log_status(f"{Color.BOLD}{Color.BLUE}Exiting after dependency installation. To run the app, provide text input.{Color.RESET}", Color.BLUE)
            sys.exit(0)

    # --- Ensure text is provided unless only building ---
    if not final_text:
        parser.error("No input text provided. Use --text, a positional argument, or pipe data to stdin.")

    # --- Logging input parameters ---
    log_status(f"{Color.BOLD}--- ParlerTTS CLI App Started ---{Color.RESET}", Color.BLUE)
    log_status(f"Input Text: '{final_text[:100]}{'...' if len(final_text) > 100 else ''}'", Color.CYAN)
    if args.output_file: 
        log_status(f"Output Path: {args.output_file}", Color.CYAN)
    log_status(f"Model Name: {args.model_name}", Color.CYAN)
    log_status(f"Model Type: {args.model_type}", Color.CYAN)
    log_status(f"Language: {args.language}", Color.CYAN)
    log_status(f"Speaker Embedding Path: {args.speaker_embedding_path or 'None (random speaker)'}", Color.CYAN)
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

        model_manager = ModelManager(model_type=args.model_type)
        tts_engine = model_manager.load_model(
            model_name=args.model_name,
            device=args.device,
            generation_config_defaults=generation_params
        )

        audio = tts_engine.generate_audio_bytes(
            text=final_text,
            language=args.language,
            speaker_embedding=speaker_embedding_tensor,
            generation_params=generation_params,
            description_prompt=args.description_prompt,
            output_filename=args.output_file
        )

        if args.output_file:
            tts_engine.save_audio_bytes(audio, args.output_file)

        if not args.quiet:
            tts_engine.play_audio_bytes(audio)
        

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
