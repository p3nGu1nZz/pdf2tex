"""Handles content blocks in PDF pages."""

import os  # Import os
import asyncio  # Import asyncio

from .utils import Utils
from .command import Command
from .environment import Environment
from .latex_text import LatexText
from .ui import console  # Import console from ui


class Block:
    """Represents a content block within a Page (text or figure)."""

    def __init__(self, bbox, parent_page, np_module, cv2_module, block_type_str, content_string):
        """Initialize a block with its bounding box, parent page, type, content, and dependencies."""
        self.parent_page = parent_page
        self.bbox = bbox
        self.np = np_module
        self.cv2 = cv2_module
        self.block = self._make_block(
            bbox, parent_page.page_img, parent_page.height
        )
        self.block_type = 0 if block_type_str == "text" else 1
        self.content_string = content_string if self.block_type == 0 else '--Block Type is Figure--'

    def _make_block(self, bbox, page_img=None, height=None):
        """Extract the block image from the page."""
        if self.np is None:
            console.print("Error: NumPy not available in Block.", style="danger")
            return None
        if height is None:
            console.print(
                "Warning: Height not provided to _make_block, "
                "potential error.",
                style="warning"
            )
            return self.np.zeros((10, 10, 3), dtype=self.np.uint8)

        if bbox.y < height and bbox.y_bottom <= height:
            return page_img[bbox.y:bbox.y_bottom, :]
        console.print(
            f"Warning: Invalid bbox coordinates - y:{bbox.y}, "
            f"y_bottom:{bbox.y_bottom}, height:{height}",
            style="warning"
        )
        return self.np.zeros((10, 10, 3), dtype=self.np.uint8)

    def generate_latex(self):
        """Generate LaTeX representation for this block."""
        if self.cv2 is None:
            console.print("Error: OpenCV not available in Block.", style="danger")
            return []
        match self.block_type:
            case 0:
                if self.content_string and not self.content_string.isspace():
                    return [
                        LatexText(self.content_string),
                        Command('vspace', arguments=['10pt'])
                    ]
                return []
            case 1:
                figure_dir = self.parent_page.parent_pdf.figures_dir
                fig_filename = (
                    f"figure_{self.parent_page.parent_pdf.num_figs}.png"
                )
                fig_path = Utils.safe_join(figure_dir, fig_filename)

                try:
                    os.makedirs(figure_dir, exist_ok=True)
                    self.cv2.imwrite(fig_path, self.block)
                    self.parent_page.parent_pdf.num_figs += 1

                    relative_fig_path = f"figures/{fig_filename}"

                    fig_env_content = [
                        Command('centering'),
                        Command(
                            'includegraphics',
                            arguments=[relative_fig_path],
                            options=[('width', r'0.8\textwidth')]
                        )
                    ]
                    return [
                        Environment(fig_env_content, 'figure', options=[('', 'H')])
                    ]
                except Exception as e:  # pylint: disable=broad-except
                    console.print(
                        f"Error saving figure {fig_path}: {e}",
                        style="danger"
                    )
                    return []
            case _:
                return []
