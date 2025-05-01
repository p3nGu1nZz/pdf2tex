"""Module for handling LaTeX file generation from PDF objects."""

import os  # Import os
from .utils import Utils
from .command import Command
from .environment import Environment
from .latex_text import LatexText  # Import LatexText
from .ui import console


class TexFile:
    """Class representing a LaTeX file with preamble and body content."""
    # Modify __init__ to accept content directly if pdf_obj is None
    def __init__(self, pdf_obj=None, content_list=None, project_dir=None, name=None, use_default_preamble=True):
        """Initialize a TexFile object from a PDF object OR a content list."""
        if pdf_obj:
            self.project_dir = pdf_obj.project_dir
            self.name = pdf_obj.name
            # If content_list is not provided, generate it (for backward compatibility/single use)
            # This path might be removed if Phase 3 is strictly enforced
            if content_list is None:
                console.print("Warning: TexFile initialized with pdf_obj but no content_list. Generating content synchronously.", style="warning")
                # This requires pdf_obj.pages to be populated and have blocks generated.
                # This synchronous path is not recommended for the main flow.
                self.content = self._generate_sync_content(pdf_obj)
            else:
                self.content = content_list  # Use provided content
        elif content_list is not None and project_dir and name:
            self.project_dir = project_dir
            self.name = name
            self.content = content_list
        else:
            raise ValueError("TexFile requires either pdf_obj or (content_list, project_dir, name)")

        self.preamble = (
            self._make_default_preamble() if use_default_preamble else []
        )

    # Helper for the less preferred synchronous path
    def _generate_sync_content(self, pdf_obj):
        content = []
        for page in pdf_obj.pages:
            for block in page.blocks:
                content.extend(block.generate_latex())
        # Add embedded images synchronously if needed (logic similar to PDF.async_generate_latex_content)
        return content

    @classmethod
    async def async_init(cls, pdf_obj=None, content_list=None, project_dir=None, name=None, use_default_preamble=True):
        """Asynchronously initialize TexFile. Primarily for Phase 3."""
        # In Phase 3, we'll call this with content_list, project_dir, name
        if content_list is not None and project_dir and name:
            return cls(content_list=content_list, project_dir=project_dir, name=name, use_default_preamble=use_default_preamble)
        # Fallback for initializing with pdf_obj (less preferred now)
        elif pdf_obj:
            console.print("Warning: TexFile.async_init called with pdf_obj. Generating content.", style="warning")
            generated_content = await pdf_obj.async_generate_latex_content()
            return cls(pdf_obj=pdf_obj, content_list=generated_content, use_default_preamble=use_default_preamble)
        else:
            raise ValueError("TexFile.async_init requires pdf_obj or (content_list, project_dir, name)")

    async def async_generate_tex_file(self, filename=None):
        """Asynchronously generate the .tex file string and write it."""
        if filename is None:
            filename = Utils.safe_join(self.project_dir, f"{self.name}.tex")

        # Ensure project directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Wrap content in document environment
        full_content = self.preamble + [Environment(self.content, 'document')]

        # Convert content objects to string
        tex_string = Utils.stringify_latex_content(full_content)

        # Pass the console instance
        lines_written = await Utils.async_write_all(filename, tex_string, console)  # Pass console
        if lines_written > 0:
            console.print(f"Wrote {lines_written} lines to {filename}", style="info")
        return lines_written

    def generate_tex_file(self, filename=None):
        """Synchronously generate the .tex file string and write it."""
        if filename is None:
            filename = Utils.safe_join(self.project_dir, f"{self.name}.tex")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        full_content = self.preamble + [Environment(self.content, 'document')]
        tex_string = Utils.stringify_latex_content(full_content)

        # Pass the console instance
        lines_written = Utils.write_all(filename, tex_string, console)  # Pass console
        if lines_written > 0:
            console.print(f"Wrote {lines_written} lines to {filename}", style="info")
        return lines_written

    @staticmethod
    def _make_default_preamble():
        """Creates a default LaTeX preamble."""
        return [
            Command('documentclass', arguments=['article']),
            Command('usepackage', arguments=['graphicx']),
            Command('usepackage', arguments=['amsmath']),
            Command('usepackage', arguments=['float']),  # Added for [H] placement
            Command('usepackage', arguments=['geometry'], options=[('margin', '1in')]),
            Command('title', arguments=['Converted Document']),
            Command('author', arguments=['pdf2tex']),
            Command('date', arguments=[r'\today']),
            LatexText('\n% --- Start of Document ---')  # Add a separator comment
        ]

    def add_to_preamble(self, obj):
        """Add a LaTeX object (Command, Environment, LatexText) to the preamble."""
        self.preamble.append(obj)
