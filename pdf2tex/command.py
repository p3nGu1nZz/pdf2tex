"""Defines the Command class for representing LaTeX commands."""


class Command:
    """Class representing a LaTeX command."""
    # pylint: disable=too-few-public-methods
    def __init__(self, command_name, arguments=None, options=None):
        arguments = arguments or []
        options = options or []

        self.command_name = command_name
        self.arguments = arguments
        self.options = options
        self.text = self._make_text()

    def _make_text(self):
        """Creates command string, formatting command name, args, and options"""
        cmd_name = self.command_name
        args = self.arguments
        opts = self.options

        text = f"\\{cmd_name}"

        arg_text = ""
        for arg in args:
            # Check if arg is another Command or similar object with a 'text' attribute
            arg_str = arg.text if hasattr(arg, 'text') else str(arg)
            arg_text += f"{{{arg_str}}}"

        if opts:
            opt_parts = []
            for opt_type, opt_val in opts:
                # Check if opt_val is another Command or similar object
                val_str = (
                    opt_val.text if hasattr(opt_val, 'text') else str(opt_val)
                )
                opt_str = f"{opt_type}={val_str}" if opt_type else val_str
                opt_parts.append(opt_str)

            opt_text = f"[{','.join(opt_parts)}]"
        else:
            opt_text = ""

        # Special handling for \begin and \end might be needed depending on usage,
        # but standard formatting is applied here.
        if cmd_name == 'begin':
            # Standard LaTeX is \begin{env}[options] but some packages might vary
            # Adjust if specific environments need different formatting
            return text + arg_text + opt_text # Common format: \begin{env} or \begin{env}[opts]
        # Default format: \command[options]{arguments}
        return text + opt_text + arg_text
