"""This module defines the Environment class, which represents a LaTeX environment."""

from .command import Command


class Environment:
    """Class representing a LaTeX environment."""
    # pylint: disable=too-few-public-methods
    def __init__(self, body, env_name, options=None):
        options = options or []

        self.env_name = env_name
        self.body = body
        self.options = options
        self.content = self._make_content()

    def _make_content(self):
        """Creates a list of LaTeX objects for the environment content."""
        env = self.env_name
        content = self.body
        opts = self.options

        start = Command('begin', arguments=[env], options=opts)
        end = Command('end', arguments=[env])
        return [start] + content + [end]
