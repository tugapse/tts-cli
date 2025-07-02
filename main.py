import argparse
import sys
import os
import numpy as np
import torch

from src.parler_tts_model import ParlerTTSModel
from src.utils import log_status
from src.colors import Color

def main():
    parser = argparse.ArgumentParser(
        description=f"{Color.BOLD}{Color.BLUE}ParlerTTS CLI App:{Color.RESET} Generate speech from text using Hugging Face Parler-TTS models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Argument ---
    parser.add_argument(
        "--text",
        type=str,
        required=True,
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
        default="parler-tts/parler-tts-mini-v1",
        help="The Hugging Face model identifier for Parler-TTS."
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

    args = parser.parse_args()

    log_status(f"{Color.BOLD}--- ParlerTTS CLI App Started ---{Color.RESET}", Color.BLUE)
    log_status(f"Input Text: '{args.text[:100]}{'...' if len(args.text) > 100 else ''}'", Color.CYAN)
    log_status(f"Output Path: {args.output_path}", Color.CYAN)
    log_status(f"Model Name: {args.model_name}", Color.CYAN)
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

        tts_engine = ParlerTTSModel(
            model_name=args.model_name,
            device=args.device,
            generation_config_defaults=generation_params
        )

        audio_bytes = tts_engine.generate_audio_bytes(
            text=args.text,
            language=args.language,
            speaker_embedding=speaker_embedding_tensor,
            generation_params=generation_params,
            description_prompt=args.description_prompt
        )

        if audio_bytes:
            output_dir = os.path.dirname(args.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                log_status(f"Created output directory: {output_dir}", Color.BLUE)
            
            with open(args.output_path, "wb") as f:
                f.write(audio_bytes)
            log_status(f"Audio successfully saved to: {args.output_path}", Color.GREEN)
        else:
            log_status(f"No audio generated for text: '{args.text}'. No file saved.", Color.YELLOW)

        log_status(f"{Color.BOLD}--- ParlerTTS CLI App Finished Successfully ---{Color.RESET}", Color.BLUE)

    except Exception as e:
        log_status(f"{Color.RED}An error occurred during execution: {e}{Color.RESET}", Color.RED)
        sys.exit(1)
    finally:
        if tts_engine:
            del tts_engine

if __name__ == "__main__":
    main()

