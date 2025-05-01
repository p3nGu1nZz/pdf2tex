"""Page object representing a PDF page containing Block objects."""

import re
import asyncio
import traceback  # Ensure traceback is imported
import gc  # Import gc
import functools  # Import functools

from .bbox import BBox
from .block import Block
from .command import Command
from .utils import Utils  # Correct the import: only import Utils
from .ui import console  # Keep console import


class Page:
    """Page object representing a PDF page containing Block objects."""

    def __init__(self, page_img, parent_pdf, np_module, cv2_module, reader_instance, executor, gpu_semaphore, page_num,
                 progress=None, page_task_id=None, batch_size=16):  # Add batch_size parameter
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
        self.batch_size = batch_size  # Store the batch_size
        # Store progress info if needed later
        self.progress = progress
        self.page_task_id = page_task_id

    async def find_blocks(self):
        """Finds bounding boxes for content blocks on the page image."""
        if self.page_img is None:
            return []
        loop = asyncio.get_running_loop()
        try:
            # Run the CPU-bound image processing in an executor
            bboxes = await loop.run_in_executor(
                self.executor, Utils.find_content_blocks, self.page_img, self.np, self.cv2, self.parent_pdf.console
            )
            return bboxes
        except Exception as e:
            console.print(f"Error finding blocks for page {self.page_num}: {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            return []

    async def process_found_blocks(self, bboxes, block_task_id):
        """Processes a list of found bounding boxes into Block objects asynchronously."""
        if not bboxes:
            self.blocks = []
            return

        loop = asyncio.get_running_loop()
        try:
            # Process each bounding box, passing the document-level block_task_id
            block_tasks = [self.process_single_block(bbox, self.page_img, block_task_id) for bbox in bboxes]
            processed_blocks = await asyncio.gather(*block_tasks)

            self.blocks = [block for block in processed_blocks if block is not None]
            del processed_blocks
            gc.collect()

        except Exception as e:
            console.print(f"Error processing found blocks for page {self.page_num}: {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            self.blocks = []
            # Don't update progress here, let the main loop handle task completion

    async def process_single_block(self, bbox, page_image, block_task_id):
        """Process a single bounding box to determine its content and create a Block."""
        loop = asyncio.get_running_loop()
        block_type_str = "figure"  # Default assumption
        content_string = ""
        block = None
        try:
            # Crop the image block
            # Ensure bbox has w and h attributes (fixed in previous step)
            y_start = max(0, bbox.y)
            y_end = min(page_image.shape[0], bbox.y + bbox.h)
            x_start = max(0, bbox.x)
            x_end = min(page_image.shape[1], bbox.x + bbox.w)

            # Check for invalid slice dimensions
            if y_start >= y_end or x_start >= x_end:
                console.print(f"Warning: Skipping block with invalid dimensions after clipping on page {self.page_num}. BBox: {bbox}", style="warning")
                return None  # Skip this block

            cropped_img = page_image[y_start:y_end, x_start:x_end]

            # Acquire semaphore before GPU-bound OCR
            async with self.gpu_semaphore:
                # Create a partial function with the arguments for readtext
                ocr_func = functools.partial(
                    self.reader.readtext,
                    cropped_img,
                    detail=0,
                    paragraph=True,
                    batch_size=self.batch_size  # Use class attribute instead of hardcoded value
                )
                # Run the partial function in the executor
                ocr_results = await loop.run_in_executor(
                    self.executor,
                    ocr_func  # Pass the partial function
                )

            # Determine block type based on OCR results
            if ocr_results:
                block_type_str = "text"
                content_string = "\n".join(ocr_results)
            # else: # Keep default 'figure' if no text found

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
                # Advance the document-level block task
                self.progress.update(block_task_id, advance=1)

            # Clean up intermediate strings
            del block_type_str, content_string, cropped_img, ocr_results, ocr_func
            gc.collect()

            return block

        except Exception as e:
            console.print(f"Error processing block on page {self.page_num} at bbox y={bbox.y}: {e}", style="danger")
            console.print(traceback.format_exc(), style="dim")
            del block_type_str, content_string, block  # Clean up
            gc.collect()
            return None

    async def async_generate_latex(self):
        """Asynchronously generate LaTeX content for all blocks on this page."""
        latex_items = []
        if not self.blocks:
            return latex_items

        # Sort blocks primarily by y-coordinate, then x-coordinate
        sorted_blocks = sorted(self.blocks, key=lambda b: (b.bbox.y, b.bbox.x))

        for block in sorted_blocks:
            try:
                latex_content = await block.async_generate_latex()
                if latex_content:
                    latex_items.append(latex_content)
            except Exception as e:
                console.print(f"Error generating LaTeX for block on page {self.page_num}: {e}", style="danger")
                console.print(traceback.format_exc(), style="dim")

        del sorted_blocks  # Clean up
        gc.collect()
        return latex_items

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

            results = self.reader.readtext(
                block_img,
                batch_size=self.batch_size,  # Use class attribute
                detail=0
            )

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

            results = self.reader.readtext(
                block_img,
                batch_size=self.batch_size,  # Use class attribute
                detail=0
            )
            return " ".join(results) if results else ""

        except Exception as e:
            console.print(f"OCR error extracting text: {e}", style="danger")
            raise RuntimeError(f"OCR error extracting text: {e}") from e
