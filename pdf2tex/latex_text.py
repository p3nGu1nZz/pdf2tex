"""Module for handling LaTeX text representation."""

from .utils import Utils


class LatexText:
    """Class representing regular LaTeX text."""
    # pylint: disable=too-few-public-methods
    def __init__(self, text):
        self.text = Utils.escape_special_chars(text)
