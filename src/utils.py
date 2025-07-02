import sys
from src.colors import Color

def log_status(message: str, color: str = Color.RESET):
    """
    Prints a status message to stderr with optional color.
    Using stderr is common for status/log output in CLI tools,
    keeping stdout clean for actual data/results if any.
    """
    print(f"{color}{message}{Color.RESET}", file=sys.stderr)

