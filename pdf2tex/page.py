"""Page object representing a PDF page containing Block objects."""

import re
import asyncio
import traceback  # Ensure traceback is imported
import gc  # Import gc

from .bbox import BBox
from .block import Block
from .command import Command
from .utils import Utils  # Correct the import: only import Utils
from .ui import console  # Keep console import


class Page:
    """Page object representing a PDF page containing Block objects."""

    def __init__(self, page_img, parent_pdf, np_module, cv2_module, reader_instance, executor, gpu_semaphore, page_num,
                 progress=None, page_task_id=None):  # Remove block_task_id
        """Initialize a page with its image, parent PDF, and dependencies."""
        self.page_img = page_img
        self.parent_pdf = parent_pdf
        self.height = page_img.shape[0] if page_img is not None else 0
        self.width = page_img.shape[1] if page_img is not None else 0
        self.np = np_module
        self.cv2 = cv2_module
        self.reader = reader_instance
        self.executor = executor
        self.gpu_semaphore = gpu_semaphore
        self.page_num = page_num
        self.blocks = []
        # Store progress info if needed later
        self.progress = progress
        self.page_task_id = page_task_id

    async def process_single_block(self, bbox, page_image, block_task_id):
        """Process a single bounding box to determine its content and create a Block."""
        loop = asyncio.get_running_loop()
        block_type_str = "figure"  # Default to figure
        content_string = ""
        block = None  # Initialize block
        try:
            # Acquire semaphore before running OCR-related tasks in executor
            async with self.gpu_semaphore:
                # Determine content type asynchronously
                block_type_str = await loop.run_in_executor(
                    self.executor, self._determine_content_type, bbox, page_image
                )

                # Extract content string based on type (if text)
                if block_type_str == "text":
                    content_string = await loop.run_in_executor(
                        self.executor, self._extract_text_from_bbox, bbox, page_image
                    )
            # Semaphore is released automatically here

            # Pass determined type and content to Block constructor
            block = Block(
                bbox=bbox,
                parent_page=self,
                np_module=self.np,
                cv2_module=self.cv2,
                block_type_str=block_type_str,
                content_string=content_string
            )

            # --- Update block progress bar ---
            if self.progress and block_task_id is not None:
                # Advance the per-page block task
                self.progress.update(block_task_id, advance=1)  # Description is set when task total is updated

            # Clean up intermediate strings
            del block_type_str, content_string
            gc.collect()  # Optional: more aggressive cleanup within block processing

            return block

        except Exception as e:
            console.print(f"Error processing block on page {self.page_num} at bbox y={bbox.y}: {e}", style="danger")
            # traceback.print_exc() # Uncomment for debugging
            # Clean up potentially partially created data
            del block_type_str, content_string, block
            gc.collect()
            return None  # Return None instead of raising

    async def async_generate_blocks(self, block_task_id):
        """Find bounding boxes and process them into Block objects asynchronously."""
        if self.page_img is None:
            self.blocks = []
            return

        loop = asyncio.get_running_loop()
        try:
            # Find bounding boxes (CPU-bound image processing)
            # Pass the console object from the parent PDF
            bboxes = await loop.run_in_executor(
                self.executor, Utils.find_content_blocks, self.page_img, self.np, self.cv2, self.parent_pdf.console  # Pass console here
            )

            num_blocks_found = len(bboxes)

            # --- Update Block Task Total ---
            if self.progress and block_task_id is not None:
                if num_blocks_found > 0:
                    self.progress.update(block_task_id, total=num_blocks_found, description=f"[yellow]Processing Blocks (Page {self.page_num})...")
                else:
                    # If no blocks, complete immediately and update description
                    self.progress.update(block_task_id, total=1, completed=1, description=f"[yellow]No Blocks Found (Page {self.page_num})")
            # --- End Update ---

            if num_blocks_found == 0:
                self.blocks = []
                return  # No blocks to process

            # Process each bounding box, passing block_task_id
            block_tasks = [self.process_single_block(bbox, self.page_img, block_task_id) for bbox in bboxes]
            processed_blocks = await asyncio.gather(*block_tasks)

            # Filter out None results from failed blocks
            self.blocks = [block for block in processed_blocks if block is not None]
            del processed_blocks  # Clean up list
            gc.collect()

        except Exception as e:
            console.print(f"Error generating blocks for page {self.page_num}: {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            self.blocks = []  # Ensure blocks is empty on error
            # Ensure progress bar is handled even on error finding blocks
            if self.progress and block_task_id is not None:
                if not self.progress.tasks[block_task_id].finished:
                    self.progress.update(block_task_id, total=1, completed=1, description=f"[red]Block Gen Error (Page {self.page_num})")

    async def async_generate_latex(self):
        """Generate LaTeX list for all blocks on this page."""
        page_content = []
        for block in self.blocks:
            try:
                latex_items = block.generate_latex()  # This is synchronous CPU-bound
                page_content.extend(latex_items)
            except Exception as e:
                console.print(f"Error generating LaTeX for block on page {self.page_num}: {e}", style="danger")
        return page_content

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
