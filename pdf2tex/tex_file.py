"""Module for handling LaTeX file generation from PDF objects."""

from .utils import Utils
from .command import Command
from .environment import Environment
from .ui import console


class TexFile:
    """Class representing a LaTeX file with preamble and body content."""
    def __init__(self, pdf_obj, use_default_preamble=True):
        """Initialize a TexFile object from a PDF object."""
        self.pdf_obj = pdf_obj
        self.preamble = (
            self._make_default_preamble() if use_default_preamble else []
        )
        self.body = []

    @classmethod
    async def async_init(cls, pdf_obj, use_default_preamble=True):
        """Asynchronous initializer for TexFile."""
        instance = cls.__new__(cls)
        instance.pdf_obj = pdf_obj
        instance.preamble = (
            cls._make_default_preamble() if use_default_preamble else []
        )
        instance.body = await pdf_obj.async_generate_latex()
        return instance

    async def async_generate_tex_file(self, filename):
        """Asynchronously generate a .tex file from the TeX objects."""
        sorted_blocks = []
        page_blocks = {}
        for page in self.pdf_obj.pages:
            if not hasattr(page, 'blocks'):
                continue
            page_num = page.page_num
            if page_num not in page_blocks:
                page_blocks[page_num] = []
            for block in page.blocks:
                if block:
                    block.page_num = page_num
                    page_blocks[page_num].append(block)

        for page_num in sorted(page_blocks.keys()):
            blocks_on_page = sorted(page_blocks[page_num], key=lambda b: b.bbox.y)
            sorted_blocks.extend(blocks_on_page)

        latex_text = []

        for command in self.preamble:
            latex_text.append(command.text)

        latex_text.append("\\begin{document}")

        for block in sorted_blocks:
            block_latex_objects = block.generate_latex()
            for latex_obj in block_latex_objects:
                if hasattr(latex_obj, 'text'):
                    latex_text.append(latex_obj.text)
                elif isinstance(latex_obj, Environment):
                    latex_text.append(latex_obj.content[0].text)
                    for body_obj in latex_obj.body:
                        if hasattr(body_obj, 'text'):
                            latex_text.append(body_obj.text)
                        else:
                            latex_text.append(str(body_obj))
                    latex_text.append(latex_obj.content[-1].text)
                else:
                    latex_text.append(str(latex_obj))

        latex_text.append("\\end{document}")

        tex_content = "\n".join(latex_text)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(tex_content)
            console.print(f"Wrote {len(latex_text)} lines to {filename}", style="success")
            return tex_content
        except (IOError, OSError) as e:
            console.print(f"Error writing tex file: {e}", style="danger")
            return None
        except Exception as e:  # pylint: disable=broad-except
            console.print(f"Unexpected error writing tex file: {e}", style="danger")
            return None

    def generate_tex_file(self, filename):
        """Write the LaTeX file content to the specified file."""
        content = self._unpack_content(self.preamble + self.body)
        Utils.write_all(filename, content, console)

    @staticmethod
    def _unpack_content(lst):
        """Convert LaTeX objects to strings suitable for writing to a file."""
        content = []
        for obj in lst:
            if isinstance(obj, Environment):
                env_content = TexFile._unpack_content(obj.content)
                content.extend(env_content)
            elif hasattr(obj, 'text'):
                content.append(obj.text)
            else:
                content.append(str(obj))
        return [
            s.replace("\x0c", Command('vspace', arguments=['10pt']).text)
            for s in content if s
        ]

    def add_to_preamble(self, obj):
        """Add a LaTeX object to the preamble."""
        self.preamble.append(obj)

    @staticmethod
    def _make_default_preamble():
        """Create default preamble for LaTeX document."""
        return [
            Command(
                'documentclass',
                arguments=['article'],
                options=[('', 'a4paper'), ('', '12pt')]
            ),
            Command('usepackage', arguments=['amsmath']),
            Command('usepackage', arguments=['amssymb']),
            Command('usepackage', arguments=['graphicx']),
            Command(
                'usepackage',
                arguments=['geometry'],
                options=[('margin', '1in')]
            ),
            Command('usepackage', arguments=['float']),
            Command('usepackage', arguments=['caption']),
            Command('setlength', arguments=[Command('parindent'), '0pt']),
            Command('setlength', arguments=[Command('parskip'), '1em']),
        ]
