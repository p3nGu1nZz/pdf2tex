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

    def __init__(self, bbox, parent_page, np_module, reader_instance, cv2_module):
        """Initialize a block with its bounding box, parent page, and dependencies."""
        self.parent_page = parent_page
        self.bbox = bbox
        self.np = np_module
        self.reader = reader_instance
        self.cv2 = cv2_module
        self.block = self._make_block(
            bbox, parent_page.page_img, parent_page.height
        )
        self.block_type, self.content_string = self._determine_content()

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

    def _determine_content(self):
        """Determine if the block is text or figure using OCR."""
        if self.reader is None:
            console.print("Error: EasyOCR Reader not available in Block.", style="danger")
            return (1, '--Block Type is Figure (OCR Not Loaded)--')
        try:
            if (
                self.block is None
                or self.block.size == 0
                or self.block.shape[0] < 5
                or self.block.shape[1] < 5
            ):
                return (1, '--Block Type is Figure (Too small)--')

            results = self.reader.readtext(self.block, paragraph=True)

            if results:
                text_parts = [result[1] for result in results]
                s = " ".join(text_parts)

                if s and not s.isspace() and len(s) > 3:
                    return (0, s)

            return (1, '--Block Type is Figure--')

        except Exception as e:  # pylint: disable=broad-except
            console.print(f"Error during OCR processing: {e}", style="danger")
            raise RuntimeError(f"OCR processing failed: {e}") from e

    async def async_determine_content(self, block_img):
        """Asynchronous version of determine_content for parallel processing."""
        if self.reader is None:
            console.print("Error: EasyOCR Reader not available in Block.", style="danger")
            return (1, '--Block Type is Figure (OCR Not Loaded)--')
        loop = asyncio.get_running_loop()
        try:
            if (
                block_img is None or block_img.size == 0 or
                block_img.shape[0] < 5 or block_img.shape[1] < 5
            ):
                return (1, '--Block Type is Figure (Too small)--')
            results = await loop.run_in_executor(
                None,
                lambda: self.reader.readtext(block_img, paragraph=True)
            )
            if results:
                text_parts = [result[1] for result in results]
                s = " ".join(text_parts)
                if s and not s.isspace() and len(s) > 3:
                    return (0, s)
            return (1, '--Block Type is Figure--')
        except Exception as e:  # pylint: disable=broad-except
            console.print(
                f"Error during async OCR processing: {e}",
                style="danger"
            )
            return (1, f'--Block Type is Figure (OCR Error: {str(e)})--')

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
