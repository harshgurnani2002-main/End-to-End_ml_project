import sys
import traceback
from colorama import Fore, Style, init

# Initialize colorama so colors work across platforms (Linux, Windows, etc.)
init(autoreset=True)


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Create a colorized detailed error message including file name, line number, type, and message.
    """
    exc_type, _, exc_tb = error_detail.exc_info()

    # Get file name & line number from traceback
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_number = 0

    return (
        f"{Fore.RED}Error occurred:{Style.RESET_ALL}\n"
        f"  File: {Fore.YELLOW}{file_name}{Style.RESET_ALL}\n"
        f"  Line: {Fore.CYAN}{line_number}{Style.RESET_ALL}\n"
        f"  Type: {Fore.MAGENTA}{exc_type.__name__ if exc_type else 'Unknown'}{Style.RESET_ALL}\n"
        f"  Message: {Fore.GREEN}{str(error)}{Style.RESET_ALL}"
    )


class CustomException(Exception):
    """
    Custom Exception that prints colorized detailed debug info.
    """
    def __init__(self, error_message: str, error_detail: sys):
        detailed_message = error_message_detail(Exception(error_message), error_detail)
        super().__init__(detailed_message)
        self.error_message = detailed_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.error_message}')"


# Example usage
if __name__ == "__main__":
    try:
        # Intentional error
        x = 1 / 0
    except Exception as e:
        raise CustomException(str(e), sys)
