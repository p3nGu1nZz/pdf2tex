"""PDF class for handling PDF documents and extracting content."""

import os
import asyncio
from .constants import DEFAULT_DATA_FOLDER, MAX_WORKERS
from .ui import console
from .utils import Utils
from .page import Page
from .environment import Environment
from .command import Command
from .latex_text import LatexText


class PDF:
    """PDF Object representing a PDF document containing Page objects."""
    def __init__(self, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                 extracted_image_paths=None, data_folder=DEFAULT_DATA_FOLDER, output_dir='.',
                 progress=None, page_task_id=None, block_task_id=None):
        """Initialize a PDF object from a file path."""
        self.path = filepath
        self.name = Utils.get_file_name(filepath)
        self.data_folder = data_folder
        # Store dependencies
        self.np = np_module
        self.cv2 = cv2_module
        self.fitz = fitz_module
        self.reader = reader_instance
        self.executor = executor
        self.console = console_instance
        self.gpu_semaphore = gpu_semaphore
        # Store progress objects
        self.progress = progress
        self.page_task_id = page_task_id
        self.block_task_id = block_task_id

        project_paths = Utils.create_latex_project_structure(output_dir, self.name)
        self.project_dir = project_paths["project_dir"]
        self.figures_dir = project_paths["figures_dir"]
        self.build_dir = project_paths["build_dir"]

        self.asset_folder = self.figures_dir

        self.temp_asset_folder = Utils.safe_join(self.build_dir, "assets")
        os.makedirs(self.temp_asset_folder, exist_ok=True)

        self.num_figs = 0
        self.pages = []
        self.num_pages = 0
        self.embedded_images = extracted_image_paths if extracted_image_paths is not None else []

    @classmethod
    async def async_init(cls, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                         extracted_image_paths=None, data_folder=DEFAULT_DATA_FOLDER, output_dir='.',
                         progress=None, page_task_id=None, block_task_id=None):
        """Asynchronous initializer for PDF class."""
        instance = cls(filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, gpu_semaphore,
                       extracted_image_paths, data_folder, output_dir,
                       progress, page_task_id, block_task_id)

        # Get page count first
        try:
            doc = await asyncio.get_running_loop().run_in_executor(instance.executor, instance.fitz.open, instance.path)
            instance.num_pages = doc.page_count
            await asyncio.get_running_loop().run_in_executor(instance.executor, doc.close)
        except Exception as e:
            instance.console.print(f"Error getting page count for {instance.name}: {e}", style="danger")
            instance.num_pages = 0

        # Convert PDF pages to Page objects (image rendering)
        instance.pages = await instance._async_pdf_to_pages()
        return instance

    def _extract_page_image(self, doc, page_num):
        """Extracts a single page image from the PDF document."""
        if self.np is None or self.cv2 is None:
            self.console.print("Error: NumPy or OpenCV not available for page extraction.", style="danger")
            return None
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_data = self.np.frombuffer(pix.samples, dtype=self.np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:  # Grayscale
                img_bgr = self.cv2.cvtColor(img_data, self.cv2.COLOR_GRAY2BGR)
            elif pix.n == 4:  # RGBA or CMYKA
                img_bgr = self.cv2.cvtColor(img_data, self.cv2.COLOR_RGBA2BGR)
            else:  # RGB or CMYK
                img_bgr = img_data  # Assume BGR if 3 channels, might need check for RGB vs BGR

            return img_bgr
        except Exception as e:
            self.console.print(f"Error extracting image for page {page_num + 1}: {e}", style="danger")
            return None

    async def _async_pdf_to_pages(self):
        """Asynchronously convert PDF file to a list of Page objects, updating progress."""
        if self.fitz is None or self.cv2 is None or self.num_pages == 0:
            return []

        loop = asyncio.get_running_loop()
        pages_list = []
        doc = None
        try:
            doc = await loop.run_in_executor(self.executor, self.fitz.open, self.path)

            # Update progress description for this PDF
            if self.progress and self.page_task_id is not None:
                self.progress.update(self.page_task_id, description=f"[cyan]Rendering Pages ({self.name})...")

            for page_num in range(self.num_pages):
                page_img = await loop.run_in_executor(
                    self.executor, self._extract_page_image, doc, page_num
                )
                if page_img is not None:
                    page_obj = Page(
                        page_img=page_img,
                        parent_pdf=self,
                        np_module=self.np,
                        cv2_module=self.cv2,
                        reader_instance=self.reader,
                        executor=self.executor,
                        gpu_semaphore=self.gpu_semaphore,
                        page_num=page_num + 1,
                        progress=self.progress,
                        page_task_id=self.page_task_id,
                        block_task_id=self.block_task_id
                    )
                    pages_list.append(page_obj)

                # Advance overall page progress after rendering each page image
                if self.progress and self.page_task_id is not None:
                    self.progress.update(self.page_task_id, advance=1)

        except Exception as e:
            self.console.print(f"Error converting PDF to pages for {self.name}: {e}", style="danger")
            return []
        finally:
            if doc:
                await loop.run_in_executor(self.executor, doc.close)

        return pages_list

    async def async_generate_latex_content(self):
        """
        Asynchronously generates the list of LaTeX content items for the PDF.
        Does NOT wrap in document environment or write file.
        Updates progress bar for block generation.
        """
        content = []

        # Update progress description
        if self.progress and self.page_task_id is not None:
            self.progress.update(self.page_task_id, description=f"[cyan]Generating Blocks ({self.name})...")

        # Generate blocks for all pages first
        block_gen_tasks = [page.async_generate_blocks(self.block_task_id) for page in self.pages]
        if block_gen_tasks:
            await asyncio.gather(*block_gen_tasks)
        else:
            self.console.print(f"No pages found or processed for {self.name} to generate blocks.", style="warning")

        # Update progress description
        if self.progress and self.page_task_id is not None:
            self.progress.update(self.page_task_id, description=f"[cyan]Generating LaTeX ({self.name})...")

        # Generate LaTeX from the blocks on each page
        latex_gen_tasks = [page.async_generate_latex() for page in self.pages]
        if latex_gen_tasks:
            results = await asyncio.gather(*latex_gen_tasks)
        else:
            self.console.print(f"No pages found or processed for {self.name} to generate LaTeX.", style="warning")
            results = []

        for page_content in results:
            content.extend(page_content)

        # Add EMBEDDED images (extracted in Phase 1)
        if self.embedded_images:
            content.append(Command('clearpage'))
            content.append(LatexText("% --- Embedded Images ---"))
            for img_path in self.embedded_images:
                try:
                    tex_file_dir = self.project_dir
                    relative_img_path = os.path.relpath(img_path, start=tex_file_dir)
                    relative_img_path = relative_img_path.replace(os.path.sep, '/')

                    figure_content = [
                        Command('centering'),
                        Command('includegraphics', arguments=[relative_img_path], options=[('width', r'0.8\textwidth')])
                    ]
                    content.append(Environment(figure_content, 'figure', options=[('', 'H')]))
                except ValueError:
                    self.console.print(f"Could not create relative path for embedded image: {img_path}", style="warning")
                except Exception as e:
                    self.console.print(f"Error processing embedded image {img_path}: {e}", style="danger")

        return content
