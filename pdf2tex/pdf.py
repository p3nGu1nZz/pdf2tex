"""PDF class for handling PDF documents and extracting content."""

import os
import asyncio
from rich.progress import Progress  # Keep this import

from .utils import Utils
from .page import Page
from .environment import Environment
from .command import Command
from .latex_text import LatexText
# Import necessary constants from constants.py
from .constants import DEFAULT_DATA_FOLDER
# Import UI elements from ui.py
from .ui import console, progress_columns


class PDF:
    """PDF Object representing a PDF document containing Page objects."""
    def __init__(self, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, data_folder=DEFAULT_DATA_FOLDER, output_dir='.'):
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
        self.console = console_instance  # Store console instance

        project_paths = Utils.create_latex_project_structure(output_dir, self.name)
        self.project_dir = project_paths["project_dir"]
        self.figures_dir = project_paths["figures_dir"]
        self.build_dir = project_paths["build_dir"]

        self.asset_folder = self.figures_dir

        self.temp_asset_folder = Utils.safe_join(self.build_dir, "assets")
        os.makedirs(self.temp_asset_folder, exist_ok=True)

        self.num_figs = 0
        self.pages = []
        self.embedded_images = []

    @classmethod
    async def async_init(cls, filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, data_folder=DEFAULT_DATA_FOLDER, output_dir='.'):
        """Asynchronous initializer for PDF class."""
        # Pass dependencies when creating instance
        instance = cls(filepath, np_module, cv2_module, fitz_module, reader_instance, executor, console_instance, data_folder, output_dir)

        if os.path.isabs(data_folder):
            instance.temp_asset_folder = os.path.join(data_folder, f"{instance.name}_assets")
        else:
            instance.temp_asset_folder = Utils.safe_join(
                os.getcwd(), data_folder, f"{instance.name}_assets"
            )
        os.makedirs(instance.temp_asset_folder, exist_ok=True)

        instance.console.print(
            f"Extracting embedded images from {instance.name}...",
            style="info"
        )
        instance.embedded_images = await Utils.extract_images_from_pdf(
            instance.path,
            instance.figures_dir,  # Save directly to figures dir
            instance.executor,
            instance.fitz,  # Pass the fitz module stored in the instance
            instance.console  # Pass the console instance stored in the instance
        )
        if instance.embedded_images:
            instance.console.print(
                f"Extracted {len(instance.embedded_images)} embedded images.",
                style="success"
            )
        else:
            instance.console.print(
                "No embedded images found or extracted.",
                style="info"
            )

        instance.pages = await instance._async_pdf_to_pages()
        return instance

    def _extract_page_image(self, doc, page_num):
        """Extracts a single page image from the PDF document."""
        if self.np is None or self.cv2 is None:
            self.console.print("Error: NumPy or OpenCV not available.", style="danger")
            return None
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = self.np.frombuffer(
                pix.samples, dtype=self.np.uint8
            ).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_GRAY2BGR)
            return img
        except Exception as e:  # pylint: disable=broad-except
            self.console.print(
                f"Error extracting image for page {page_num}: {e}",
                style="danger"
            )
            return None

    async def _async_pdf_to_pages(self):
        """Asynchronously convert PDF file to a list of Page objects."""
        if self.fitz is None or self.cv2 is None:
            self.console.print("Error: PyMuPDF or OpenCV not available.", style="danger")
            return []
        loop = asyncio.get_running_loop()
        try:
            doc = self.fitz.open(self.path)
            num_pages = len(doc)
            if num_pages == 0:
                self.console.print("PDF has no pages.", style="warning")
                doc.close()
                return []

            # Use Progress context manager
            # Remove transient=True for debugging
            with Progress(*progress_columns, console=self.console, transient=False) as progress:
                # --- Task 1: Extract Page Images ---
                page_extract_task = progress.add_task("[status]Extracting page images...", total=num_pages)
                futures = [
                    loop.run_in_executor(self.executor, self._extract_page_image, doc, i)
                    for i in range(num_pages)
                ]
                page_images = []
                for i, future in enumerate(asyncio.as_completed(futures)):
                    img = await future
                    page_images.append((i, img))  # Store index with image
                    progress.update(page_extract_task, advance=1)

                # Sort images by original page number
                page_images.sort(key=lambda x: x[0])
                doc.close()  # Close doc after image extraction

                # Filter out None images before creating Page objects
                valid_page_data = [(i, img) for i, img in page_images if img is not None]
                if not valid_page_data:
                    self.console.print("No valid page images extracted.", style="warning")
                    return []

                # Create Page objects, passing dependencies
                pages = [
                    Page(img, self, self.np, self.cv2, self.reader, self.executor, i)
                    for i, img in valid_page_data
                ]

                # --- Task 2: Generate Content Blocks ---
                block_gen_task = progress.add_task("[status]Generating content blocks...", total=len(pages))
                block_gen_futures = [page.async_generate_blocks() for page in pages]
                for future in asyncio.as_completed(block_gen_futures):
                    try:
                        await future  # Wait for block generation to complete or raise error
                        progress.update(block_gen_task, advance=1)
                    except Exception as block_gen_error:
                        # Error is likely already printed in Page.async_generate_blocks
                        self.console.print(f"Halting due to error during block generation.", style="critical")
                        # Stop other tasks if desired (complex with as_completed)
                        # Re-raise the exception to stop PDF processing
                        raise block_gen_error

            # Progress bar will now persist after completion
            self.console.print("Finished page processing.", style="info")
            return pages

        except Exception as e:
            self.console.print(f"Error converting PDF to pages: {e}", style="danger")
            import traceback
            traceback.print_exc()
            # Ensure doc is closed in case of error
            if 'doc' in locals() and doc and not doc.is_closed:
                doc.close()
            return []  # Return empty list on error

    async def async_generate_latex(self):
        """Asynchronously generate LaTeX content for the PDF."""
        content = []
        self.console.print("Generating LaTeX asynchronously...", style="status")

        tasks = [page.async_generate_latex() for page in self.pages]

        if tasks:
            with Progress(*progress_columns, console=self.console, transient=True) as progress:
                task = progress.add_task("Processing pages", total=len(tasks))
                results = []
                for coro in asyncio.as_completed(tasks):
                    page_content = await coro
                    results.append(page_content)
                    progress.update(task, advance=1)
        else:
            results = []

        for page_content in results:
            content.extend(page_content)

        if self.embedded_images:
            content.append(Command('section', arguments=['Embedded Images']))
            content.append(
                LatexText('The following images were embedded in the PDF:')
            )
            content.append(Command('vspace', arguments=['10pt']))

            for idx, img_path in enumerate(self.embedded_images):
                img_filename = os.path.basename(img_path)
                fig_env_content = [
                    Command('centering'),
                    Command(
                        'includegraphics',
                        arguments=[f"figures/{img_filename}"],
                        options=[('width', r'0.8\textwidth')]
                    ),
                    Command('caption', arguments=[f"Embedded image {idx+1}"])
                ]
                content.append(
                    Environment(fig_env_content, 'figure', options=[('', 'H')])
                )

        return [Environment(content, 'document')]
