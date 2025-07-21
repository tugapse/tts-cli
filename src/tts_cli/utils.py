import sys


class Color:
    """ANSI escape codes for colored console output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
    WARNING = '\033[93m'



def log_status(message: str, color: str = Color.RESET):
    """
    Prints a status message to stderr with optional color.
    Using stderr is common for status/log output in CLI tools,
    keeping stdout clean for actual data/results if any.
    """
    print(f"{color}{message}{Color.RESET}", file=sys.stderr)

