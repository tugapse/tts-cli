# model_manager.py

from src.utils import Color , log_status


class ModelManager:
    """
    Manages the loading of different TTS models based on the model type.
    """
    def __init__(self, model_type):
        self.model_type = model_type
        log_status(f"ModelManager initialized for type: {self.model_type}", Color.CYAN)

    def load_model(self, model_name, device, generation_config_defaults):
        """
        Loads and returns the appropriate TTS model instance.
        """
        if self.model_type == "parler":

            from src.models.parler_tts_model import ParlerTTSModel
            
            log_status(f"Loading ParlerTTS model engine: {model_name} on device {device}...", Color.BLUE)
            return ParlerTTSModel(
                model_name=model_name,
                device=device,
                generation_config_defaults=generation_config_defaults
            )
        elif self.model_type == "orpheus":

            from src.models.orpheus_tts_model import OrpheusTTSModel
            
            log_status(f"Loading OrpheusTTS model engine: {model_name} on device {device}...", Color.BLUE)
            return OrpheusTTSModel(
                model_name=model_name,
                device=device,
                generation_config_defaults=generation_config_defaults
            )
        # Add more model types here in the future if needed
        # elif self.model_type == "another_tts_model":
        #     log_status(f"Loading AnotherTTSModel: {model_name} on device {device}...", Color.BLUE)
        #     return AnotherTTSModel(model_name, device, generation_config_defaults)
        else:
            raise ValueError(f"{Color.RED}Unsupported model type: '{self.model_type}'. Please choose a supported type like 'parlertts'.{Color.RESET}")

