"""Page object representing a PDF page containing Block objects."""

import re
import asyncio
import traceback  # Import traceback

from .bbox import BBox
from .block import Block
from .command import Command
from .utils import Utils  # Correct the import: only import Utils
from .ui import console  # Keep console import


class Page:
    """Page object representing a PDF page containing Block objects."""

    def __init__(self, page_img, parent_pdf, np_module, cv2_module, reader_instance, executor, page_num):
        """Initialize a page with its image, parent PDF, and dependencies."""
        self.page_img = page_img
        self.parent_pdf = parent_pdf
        self.height = page_img.shape[0] if page_img is not None else 0
        self.width = page_img.shape[1] if page_img is not None else 0
        self.np = np_module
        self.cv2 = cv2_module
        self.reader = reader_instance
        self.executor = executor
        self.page_num = page_num
        self.blocks = []

    async def process_single_block(self, bbox, page_image):
        """Process a single bounding box to determine its content and create a Block."""
        loop = asyncio.get_running_loop()
        try:
            # Determine content type asynchronously
            block_type_str = await loop.run_in_executor(
                self.executor, self._determine_content_type, bbox, page_image
            )

            # Extract content string based on type
            content_string = ""
            if block_type_str == "text":
                content_string = await loop.run_in_executor(
                    self.executor, self._extract_text_from_bbox, bbox, page_image
                )
            # Note: Image content string (filename) is handled within Block.generate_latex

            # Directly create the Block instance, passing all dependencies
            block = Block(
                bbox=bbox,
                parent_page=self,
                np_module=self.np,
                reader_instance=self.reader,
                cv2_module=self.cv2
            )
            # The block type and content string are determined within Block.__init__ now

            return block

        except Exception as e:
            console.print(f"Error processing block on page {self.page_num + 1} at bbox y={bbox.y}: {e}", style="danger")
            raise  # Re-raise the exception to stop processing

    async def async_generate_blocks(self):
        """Asynchronously generate blocks for the page."""
        try:
            if self.page_img is None:
                console.print(f"Warning: Page {self.page_num + 1} image is None, skipping block generation.", style="warning")
                return

            bboxes = self._get_bounding_boxes(self.page_img)
            if not bboxes:
                console.print(f"No content blocks found on page {self.page_num + 1}.", style="info")
                return

            tasks = [self.process_single_block(bbox, self.page_img) for bbox in bboxes]
            generated_blocks = await asyncio.gather(*tasks)
            self.blocks = [block for block in generated_blocks if block]

        except Exception as e:
            if not isinstance(e, TypeError) or "Block.__init__" not in str(e):
                console.print(f"Error during block generation setup for page {self.page_num + 1}: {e}", style="danger")
                traceback.print_exc()
            raise

    async def async_generate_latex(self):
        """Asynchronously generate LaTeX content for the page."""
        content = []
        for block in self.blocks:
            content.extend(block.generate_latex())
        return content + [
            Command('par'),
            Command('vspace', arguments=['10pt'])
        ]

    def _load_page_image(self):
        """Returns the page image that was already loaded during PDF processing."""
        return self.page_img

    def _get_bounding_boxes(self, page_image):
        """Get bounding boxes for the page image."""
        return Utils.find_content_blocks(page_image, self.cv2, self.np, console, BBox_class=BBox)

    def _determine_content_type(self, bbox, page_image):
        """Determines if a bbox contains text or image using OCR."""
        if self.reader is None:
            console.print("Error: EasyOCR Reader not initialized.", style="danger")
            raise RuntimeError("EasyOCR Reader not initialized.")
        try:
            block_img = Utils.extract_block_image(bbox, page_image, self.height, self.np, console)

            if (
                block_img is None
                or block_img.size == 0
                or block_img.shape[0] < 5
                or block_img.shape[1] < 5
            ):
                return "image"

            results = self.reader.readtext(block_img, paragraph=True, detail=0)

            if results and any(s and not s.isspace() for s in results):
                return "text"

            return "image"

        except Exception as e:
            console.print(f"OCR error determining content type: {e}", style="danger")
            raise RuntimeError(f"OCR error determining content type: {e}") from e

    def _extract_text_from_bbox(self, bbox, page_image):
        """Extracts text using OCR from a given bbox."""
        if self.reader is None:
            console.print("Error: EasyOCR Reader not initialized.", style="danger")
            raise RuntimeError("EasyOCR Reader not initialized.")
        try:
            block_img = Utils.extract_block_image(bbox, page_image, self.height, self.np, console)

            if (
                block_img is None
                or block_img.size == 0
                or block_img.shape[0] < 5
                or block_img.shape[1] < 5
            ):
                return ""

            results = self.reader.readtext(block_img, paragraph=True, detail=0)
            return " ".join(results) if results else ""

        except Exception as e:
            console.print(f"OCR error extracting text: {e}", style="danger")
            raise RuntimeError(f"OCR error extracting text: {e}") from e
