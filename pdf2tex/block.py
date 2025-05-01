"""Handles content blocks in PDF pages."""

import os
import asyncio
import traceback

from .utils import Utils
from .command import Command
from .latex_text import LatexText
from .environment import Environment
from .ui import console


class Block:
    """Represents a content block (text or figure) within a PDF page."""

    def __init__(self, bbox, parent_page, np_module, cv2_module, block_type_str, content_string):
        """Initialize a Block object."""
        self.bbox = bbox
        self.parent_page = parent_page
        self.np = np_module
        self.cv2 = cv2_module
        self.block_type = block_type_str  # 'text' or 'figure'
        self.content = content_string  # Raw text or placeholder for figure path

    async def async_generate_latex(self):
        """Asynchronously generates the LaTeX string for this block."""
        loop = asyncio.get_running_loop()
        try:
            if self.block_type == "text":
                # Simple text block
                return await loop.run_in_executor(None, str, LatexText(self.content))
            elif self.block_type == "figure":
                # Figure handling - Needs refinement based on how figure paths are determined
                figure_rel_path = await self._get_figure_path()  # Placeholder for async path logic

                if not figure_rel_path:
                    # Fallback to text if figure path fails
                    console.print(f"Warning: Could not determine figure path for block at {self.bbox}. Treating as text.", style="warning")
                    return await loop.run_in_executor(None, str, LatexText(f"% FIGURE PLACEHOLDER (Path Error): {self.bbox}\n"))

                caption = f"Figure caption placeholder"  # TODO: Generate better caption/label
                label = f"fig:{Utils.sanitize_filename(os.path.basename(figure_rel_path))}"

                figure_content = [
                    Command('centering'),
                    Command('includegraphics', options={'width': r'0.8\textwidth'}, arguments=[figure_rel_path]),
                    Command('caption', arguments=[caption]),
                    Command('label', arguments=[label])
                ]
                env = Environment(figure_content, 'figure', options=['H'])
                return await loop.run_in_executor(None, str, env)
            else:
                # Fallback for unknown types
                return await loop.run_in_executor(None, str, LatexText(f"% Unknown block type: {self.block_type}\n{self.content}"))

        except Exception as e:
            console.print(f"Error generating LaTeX for block: {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            return await loop.run_in_executor(None, str, LatexText(f"% Error generating block: {e}\n"))

    async def _get_figure_path(self):
        """
        Determines and potentially saves the image for a figure block.
        Returns the relative path to be used in LaTeX.
        """
        # 1. Get the image data for this block's bbox from the parent page
        page_img = self.parent_page.page_img
        if page_img is None:
            return None

        y_start = max(0, self.bbox.y)
        y_end = min(page_img.shape[0], self.bbox.y2)
        x_start = max(0, self.bbox.x)
        x_end = min(page_img.shape[1], self.bbox.x2)

        if y_start >= y_end or x_start >= x_end:
            return None  # Invalid dimensions

        block_image_data = page_img[y_start:y_end, x_start:x_end]

        # 2. Define filename and path
        # Use the assets_dir stored in the parent PDF object
        assets_dir = self.parent_page.parent_pdf.assets_dir  # Use correct path
        os.makedirs(assets_dir, exist_ok=True)

        # Create a unique filename (e.g., based on page and bbox)
        img_filename = f"page_{self.parent_page.page_num}_block_{self.bbox.y}_{self.bbox.x}.png"
        img_full_path = Utils.safe_join(assets_dir, img_filename)

        # 3. Save the image asynchronously using cv2 in executor
        loop = asyncio.get_running_loop()
        try:
            def save_image():
                return self.cv2.imwrite(img_full_path, block_image_data)

            success = await loop.run_in_executor(self.parent_page.executor, save_image)

            if not success:
                console.print(f"Error saving block image: {img_full_path}", style="danger")
                return None

            # 4. Return the relative path for LaTeX (relative to the .tex file in project_dir)
            relative_path = os.path.join("assets", img_filename).replace("\\", "/")
            return relative_path

        except Exception as e:
            console.print(f"Error during block image saving ({img_full_path}): {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            return None

    def __repr__(self):
        """String representation for debugging."""
        return f"Block({self.block_type}, bbox={self.bbox})"
