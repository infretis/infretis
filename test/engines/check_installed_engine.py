import shutil
import sys

def detect_program(programs : list[str]) -> str:
    # List of possible program names

    for program in programs:
        # Check if the program exists in the system's PATH
        if shutil.which(program):
            return program  # Return the program name for further use

    # return empty string if no programs are found
    return ""
