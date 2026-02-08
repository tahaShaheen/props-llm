import os
import sys

_COLOR_CODES = {
    "red": "31",
    "yellow": "33",
    "green": "32",
    "blue": "34",
    "gray": "90",
}


def colorize(text, color):
    if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
        return text
    code = _COLOR_CODES.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(text):
    return colorize(text, "red")


def yellow(text):
    return colorize(text, "yellow")


def green(text):
    return colorize(text, "green")


def blue(text):
    return colorize(text, "blue")


def gray(text):
    return colorize(text, "gray")
