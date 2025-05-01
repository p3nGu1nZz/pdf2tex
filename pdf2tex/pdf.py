"""PDF class for handling PDF documents and extracting content."""

import os
import asyncio
import traceback
from .constants import DEFAULT_DATA_FOLDER
from .ui import console
from .utils import Utils
from .page import Page
from .environment import Environment
from .command import Command
from .latex_text import LatexText
from .references import ReferenceExtractor


class PDF:
    """PDF Object representing a PDF document containing Page objects."""

    def __init__(self, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                 extracted_image_paths=None, data_folder=DEFAULT_DATA_FOLDER, output_dir='.',
                 progress=None, page_task_id=None, batch_size=16):  # Add batch_size parameter
        """Initialize PDF object."""
        self.filepath = filepath
        self.name = Utils.get_file_name(filepath)
        self.np = np_module
        self.cv2 = cv2_module
        self.fitz = fitz_module
        self.reader = reader_instance
        self.executor = executor
        self.console = console_instance
        self.gpu_semaphore = gpu_semaphore
        self.data_folder = data_folder
        self.output_dir = os.path.abspath(output_dir)
        self.assets_dir = Utils.safe_join(self.output_dir, "assets")
        self.progress = progress
        self.page_task_id = page_task_id
        self.pages = []
        self.num_pages = 0
        self.embedded_images = extracted_image_paths if extracted_image_paths else []
        self.batch_size = batch_size  # Store for use in page creation

    @classmethod
    async def async_init(cls, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                         extracted_image_paths=None, data_folder=DEFAULT_DATA_FOLDER, output_dir='.',
                         progress=None, page_task_id=None, batch_size=16):  # Add batch_size parameter
        """Asynchronous initializer for PDF class."""
        instance = cls(filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                       extracted_image_paths, data_folder, output_dir,
                       progress, page_task_id, batch_size)  # Pass batch_size
        try:
            loop = asyncio.get_running_loop()
            doc = await loop.run_in_executor(executor, fitz_module.open, filepath)
            instance.num_pages = doc.page_count
            await loop.run_in_executor(executor, doc.close)

            if instance.num_pages > 0:
                pass
            else:
                console_instance.print(f"Warning: PDF '{instance.name}' has 0 pages.", style="warning")

        except Exception as e:
            console_instance.print(f"Error initializing PDF '{instance.name}': {e}", style="danger")
            console_instance.print(traceback.format_exc(), style="dim")
            return None

        return instance

    async def _extract_page_image(self, page_num, doc):
        """Extracts the image of a single page."""
        if self.np is None or self.cv2 is None:
            self.console.print("Error: NumPy or OpenCV not available for page extraction.", style="danger")
            return None
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_data = self.np.frombuffer(pix.samples, dtype=self.np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img_bgr = self.cv2.cvtColor(img_data, self.cv2.COLOR_GRAY2BGR)
            elif pix.n == 4:
                img_bgr = self.cv2.cvtColor(img_data, self.cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img_data

            return img_bgr
        except Exception as e:
            self.console.print(f"Error extracting image for page {page_num + 1}: {e}", style="danger")
            return None

    async def _async_pdf_to_pages(self, progress=None, page_task_id=None):
        """Converts PDF pages to Page objects containing images."""
        pass

    async def async_generate_latex_content(self):
        """
        Asynchronously generates LaTeX content items and BibTeX entries.
        Returns tuple: (list_of_body_content_items, list_of_bibtex_strings)
        """
        body_content = []
        bibtex_entries = []
        block_task_id = None
        total_blocks_in_doc = 0
        page_bboxes = {}

        if not self.pages and self.num_pages > 0:
            doc = None
            try:
                loop = asyncio.get_running_loop()
                doc = await loop.run_in_executor(self.executor, self.fitz.open, self.filepath)
                page_creation_tasks = []
                for i in range(self.num_pages):
                    page_creation_tasks.append(self._create_page_object(i, doc))
                self.pages = await asyncio.gather(*page_creation_tasks)
            except Exception as e:
                self.console.print(f"Error creating page objects for {self.name}: {e}", style="danger")
                return [], []
            finally:
                if doc:
                    await loop.run_in_executor(self.executor, doc.close)

        if self.progress:
            block_task_id = self.progress.add_task(f"[yellow]Finding Blocks ({self.name})...", total=None, start=True)

        try:
            find_tasks = []
            for page in self.pages:
                find_tasks.append(page.find_blocks())

            all_page_bboxes_results = await asyncio.gather(*find_tasks)

            for i, page in enumerate(self.pages):
                bboxes = all_page_bboxes_results[i]
                page_bboxes[page.page_num] = bboxes
                total_blocks_in_doc += len(bboxes)

            if self.progress and block_task_id is not None:
                if total_blocks_in_doc > 0:
                    self.progress.update(block_task_id, total=total_blocks_in_doc, description=f"[yellow]Processing Blocks ({self.name})...")
                else:
                    self.progress.update(block_task_id, total=1, completed=1, description=f"[yellow]No Blocks Found ({self.name})")
                    self.progress.stop_task(block_task_id)
                    self.progress.remove_task(block_task_id)
                    block_task_id = None

            if total_blocks_in_doc == 0:
                return body_content, bibtex_entries

            process_tasks = []
            for page in self.pages:
                process_tasks.append(page.process_found_blocks(page_bboxes.get(page.page_num, []), block_task_id))
            await asyncio.gather(*process_tasks)

            reference_section_started = False
            reference_blocks_text = []

            for page in self.pages:
                sorted_blocks = sorted(page.blocks, key=lambda b: (b.bbox.y, b.bbox.x))

                for block in sorted_blocks:
                    if block.block_type == "text" and block.content.strip():
                        if not reference_section_started:
                            reference_section_started = ReferenceExtractor.identify_reference_section(block.content)

                        if reference_section_started:
                            reference_blocks_text.append(block.content)
                            body_content.append(LatexText(f"% Reference block found on page {page.page_num}\n"))
                        else:
                            latex_item = await block.async_generate_latex()
                            if latex_item:
                                body_content.append(latex_item)
                    elif block.block_type == "figure" or not reference_section_started:
                        latex_item = await block.async_generate_latex()
                        if latex_item:
                            body_content.append(latex_item)

                del sorted_blocks

            if reference_blocks_text:
                combined_ref_text = "\n".join(reference_blocks_text)
                extracted_bibs = ReferenceExtractor.extract_bibtex_entries(combined_ref_text)
                bibtex_entries.extend(extracted_bibs)

        except Exception as e:
            self.console.print(f"Error during content generation for {self.name}: {e}", style="danger")
            self.console.print(traceback.format_exc(), style="dim")
            if self.progress and block_task_id is not None and not self.progress.tasks[block_task_id].finished:
                self.progress.update(block_task_id, description=f"[red]Block Processing Error ({self.name})")

        finally:
            if self.progress and block_task_id is not None:
                task = self.progress.tasks[block_task_id]
                if not task.finished:
                    self.progress.update(block_task_id, completed=task.total)
                self.progress.stop_task(block_task_id)
                self.progress.remove_task(block_task_id)
            del page_bboxes

        embedded_images = self._collect_embedded_images()
        body_content.extend(embedded_images)

        return body_content, bibtex_entries

    async def _create_page_object(self, page_index, doc):
        """Helper to create a Page object asynchronously."""
        page_img = await self._extract_page_image(page_index, doc)
        if page_img is None:
            self.console.print(f"Warning: Failed to extract image for page {page_index + 1} of {self.name}", style="warning")
        return Page(
            page_img=page_img,
            parent_pdf=self,
            np_module=self.np,
            cv2_module=self.cv2,
            reader_instance=self.reader,
            executor=self.executor,
            gpu_semaphore=self.gpu_semaphore,
            page_num=page_index + 1,
            progress=self.progress,
            page_task_id=self.page_task_id,
            batch_size=self.batch_size  # Pass batch_size to Page
        )

    def _collect_embedded_images(self):
        """Generate LaTeX for embedded images found in Phase 1."""
        latex_items = []
        if not self.embedded_images:
            return latex_items

        self.console.print(f"Processing {len(self.embedded_images)} embedded images for {self.name}...", style="info")

        for img_path in self.embedded_images:
            try:
                img_filename = os.path.basename(img_path)
                final_img_path_abs = Utils.safe_join(self.assets_dir, img_filename)
                os.makedirs(self.assets_dir, exist_ok=True)
                relative_path = os.path.join("assets", img_filename).replace("\\", "/")
                caption = f"Embedded image: {Utils.escape_special_chars(img_filename)}"
                label = f"fig:embedded_{Utils.sanitize_filename(img_filename)}"
                figure_content = [
                    Command('centering'),
                    Command('includegraphics', options={'width': r'0.8\textwidth'}, arguments=[relative_path]),
                    Command('caption', arguments=[caption]),
                    Command('label', arguments=[label])
                ]
                env = Environment(figure_content, 'figure', options=['H'])
                latex_items.append(env)
            except Exception as e:
                self.console.print(f"Error processing embedded image {img_path}: {e}", style="danger")

        return latex_items
