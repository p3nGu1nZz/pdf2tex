# pylint: disable=too-many-lines
# pylint: disable=no-member
# pylint: disable=import-outside-toplevel, global-statement
"""
Core conversion logic for PDF to LaTeX.

Contains the async_convert and convert functions, along with helper
coroutines for different phases of the conversion process.
"""

import os
import sys
import asyncio
import concurrent.futures
import time
import traceback
import gc

from rich.progress import Progress

from .utils import Utils
from .bbox import BBox
from .tex_file import TexFile
from .pdf import PDF
from .ui import console, progress_columns, custom_theme
from .constants import (
    DEFAULT_DATA_FOLDER,
    MIN_TEXT_SIZE,
    HORIZONTAL_POOLING
)

# --- Globals that will be initialized later ---
READER = None
CV2 = None
FITZ = None
PLT = None
NP = None
TORCH = None
EASYOCR = None
IS_LOADED = False

# Ensure TORCH is imported if not already
try:
    import torch as TORCH
except ImportError:
    TORCH = None  # Handle case where torch might not be installed initially

# --- Dependency Loading Function ---
def _ensure_dependencies_loaded():
    """Loads heavy dependencies and initializes READER if not already done."""
    global READER, CV2, FITZ, PLT, NP, TORCH, EASYOCR, IS_LOADED

    # Skip if already loaded
    if IS_LOADED:
        return

    try:
        import cv2 as cv2_module
        import fitz as fitz_module
        import matplotlib.pyplot as plt_module
        import numpy as np_module
        import torch as torch_module
        import easyocr as easyocr_module

        CV2 = cv2_module
        FITZ = fitz_module
        PLT = plt_module
        NP = np_module
        TORCH = torch_module
        EASYOCR = easyocr_module

        # Set torch num_threads to 1 to avoid oversubscription
        TORCH.set_num_threads(1)

        # Disable CUDA multi-threading if using GPU
        if TORCH.cuda.is_available():
            TORCH.cuda.set_device(0)

        console.print("Initializing EasyOCR Reader...", style="info")
        READER = EASYOCR.Reader(['en'], gpu=TORCH.cuda.is_available(), quantize=True)

        # Call flatten_parameters on LSTM modules to fix the warning
        _flatten_lstm_parameters(READER)

        console.print("EasyOCR Reader initialized.", style="success")
        IS_LOADED = True

    except ImportError as e:
        console.print(f"Error importing dependencies: {e}", style="danger")
        console.print("Please ensure all required libraries (OpenCV, PyMuPDF, Matplotlib, NumPy, PyTorch, EasyOCR) are installed.", style="warning")
        sys.exit(1)


def _flatten_lstm_parameters(reader):
    """Flatten parameters for any LSTM modules in the reader to fix the warning."""
    if not hasattr(reader, 'model'):
        return

    # Recursively find and flatten LSTM modules
    def find_and_flatten_lstm(module):
        for child in module.children():
            if isinstance(child, TORCH.nn.LSTM):
                child.flatten_parameters()
            elif len(list(child.children())) > 0:
                find_and_flatten_lstm(child)

    # Apply to detection model
    if hasattr(reader, 'detector') and hasattr(reader.detector, 'model'):
        find_and_flatten_lstm(reader.detector.model)

    # Apply to recognition model
    if hasattr(reader, 'recognizer') and hasattr(reader.recognizer, 'model'):
        find_and_flatten_lstm(reader.recognizer.model)


# --- Print Helpers ---
def print_status(msg):
    """Print status messages with rich colored output."""
    console.print(msg, style="status")


def print_success(msg):
    """Print success messages with rich colored output."""
    console.print(msg, style="success")


def print_error(msg):
    """Print error messages with rich colored output."""
    console.print(msg, style="danger")


async def _extract_images_for_pdf(pdf_path, base_output_dir, executor, fitz_module, console_instance, progress, task_id):
    """Helper coroutine for Phase 1: Extracts images for a single PDF."""
    pdf_name = Utils.get_file_name(pdf_path)
    project_paths = Utils.create_latex_project_structure(base_output_dir, pdf_name)
    build_dir = project_paths["build_dir"]
    temp_asset_folder = Utils.safe_join(build_dir, "assets")
    os.makedirs(temp_asset_folder, exist_ok=True)

    progress.update(task_id, description=f"[cyan]Extracting {pdf_name}...")

    success, image_paths = await Utils.extract_images_from_pdf(
        pdf_path, temp_asset_folder, executor, fitz_module, console_instance, progress, task_id
    )
    if not success:
        progress.update(task_id, description=f"[red]Failed {pdf_name}")
    return pdf_path, project_paths["project_dir"], success, image_paths


async def _process_pdf_content(pdf_path, project_dir, image_paths, gpu_executor, gpu_semaphore, data_folder, progress, page_task_id):
    """Helper coroutine for Phase 2: Processes content and generates LaTeX."""
    pdf_name = Utils.get_file_name(pdf_path)
    pdf = None
    try:
        pdf = await PDF.async_init(
            pdf_path, NP, CV2, FITZ, READER, gpu_executor, console, gpu_semaphore,
            extracted_image_paths=image_paths,
            data_folder=data_folder,
            output_dir=os.path.dirname(project_dir),
            progress=progress,
            page_task_id=page_task_id
        )
        if not pdf or not pdf.pages:
            console.print(f"[Phase 2] Failed to initialize or find pages for {pdf_name}", style="danger")
            return False, project_dir, pdf_name, []

        generated_content = await pdf.async_generate_latex_content()

        return True, project_dir, pdf_name, generated_content

    except Exception as e:
        console.print(f"[Phase 2] Error processing content for {pdf_name}: {e}", style="danger")
        console.print(traceback.format_exc(), style="dim")
        if pdf and pdf.num_pages > 0 and progress and page_task_id is not None:
            task = progress.tasks[page_task_id]
            remaining_pages = pdf.num_pages
            progress.update(page_task_id, advance=remaining_pages, description="[red]Page processing error")
        return False, project_dir, pdf_name, []


async def _assemble_latex_file(project_dir, pdf_name, content_list, progress, assembly_task_id):
    """Helper coroutine for Phase 3: Assembles and writes the .tex file."""
    try:
        progress.update(assembly_task_id, description=f"[yellow]Assembling {pdf_name}.tex...")
        tex_file = await TexFile.async_init(content_list=content_list, project_dir=project_dir, name=pdf_name)
        await tex_file.async_generate_tex_file()
        progress.update(assembly_task_id, advance=1, description=f"[yellow]Assembling {pdf_name}.tex... Done")
        return True
    except Exception as e:
        console.print(f"[Phase 3] Error assembling file for {pdf_name}: {e}", style="danger")
        progress.update(assembly_task_id, description=f"[red]Assembly failed {pdf_name}")
        return False


async def async_convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER, max_workers=1, image_workers=4):
    """Asynchronously convert PDF(s) using a three-phase approach with progress bars."""
    _ensure_dependencies_loaded()
    start_time = time.monotonic()

    pdf_paths_to_process = []
    if os.path.isdir(source_path):
        pdf_files = [f for f in os.listdir(source_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            console.print(f"No PDF files found in directory: {source_path}", style="warning")
            return
        pdf_paths_to_process = [os.path.join(source_path, f) for f in pdf_files]
        console.print(f"Found {len(pdf_paths_to_process)} PDF(s) in directory.", style="info")
    elif os.path.isfile(source_path) and source_path.lower().endswith('.pdf'):
        pdf_paths_to_process = [source_path]
        console.print(f"Processing single PDF file: {source_path}", style="info")
    else:
        print_error(f"Invalid source path: {source_path}")
        return

    if not pdf_paths_to_process:
        console.print("No PDF files found to process.", style="warning")
        return

    # Initialize variables to store summary counts
    successful_extractions_count = 0
    failed_extractions_count = 0
    successful_processing_count = 0
    failed_processing_count = 0
    assembly_success_count = 0
    failed_assembly_count = 0
    total_pdfs_processed = len(pdf_paths_to_process)  # Store initial count

    with Progress(*progress_columns, console=console, transient=False) as progress:

        phase1_task_id = progress.add_task("[bold cyan]Phase 1: Extracting Images...", total=len(pdf_paths_to_process))
        phase1_start_time = time.monotonic()
        image_executor = concurrent.futures.ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix='ImgExtract')
        extraction_tasks = []
        pdf_extraction_task_ids = {}

        for pdf_path in pdf_paths_to_process:
            task_id = progress.add_task(f"[cyan]Queued {Utils.get_file_name(pdf_path)}", total=1, start=False)
            pdf_extraction_task_ids[pdf_path] = task_id
            extraction_tasks.append(
                _extract_images_for_pdf(pdf_path, output_dir, image_executor, FITZ, console, progress, task_id)
            )

        extraction_results = []
        if extraction_tasks:
            extraction_results = await asyncio.gather(*extraction_tasks)

        image_executor.shutdown(wait=True)
        progress.update(phase1_task_id, completed=len(pdf_paths_to_process), description="[bold green]Phase 1: Image Extraction Complete")
        # Calculate counts but don't print yet
        successful_extractions = [res for res in extraction_results if res[2]]
        successful_extractions_count = len(successful_extractions)
        failed_extractions_count = total_pdfs_processed - successful_extractions_count

        if not successful_extractions:
            # Print immediate exit message if needed
            console.print("No PDFs remaining after image extraction phase. Exiting.", style="warning")
            return  # Exit before printing summaries

        phase2_docs_task_id = progress.add_task("[bold magenta]Phase 2: Processing Documents...", total=successful_extractions_count)
        total_pages_to_process = 0
        page_count_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='PageCount')
        temp_doc = None
        console.print("Calculating total pages for progress...", style="info")
        for pdf_path, _, _, _ in successful_extractions:
            temp_doc = None  # Initialize temp_doc for each iteration
            try:
                # Open doc and get count within the try block
                temp_doc = await asyncio.get_running_loop().run_in_executor(page_count_executor, FITZ.open, pdf_path)
                total_pages_to_process += temp_doc.page_count
            except Exception as e:
                console.print(f"Warning: Could not get page count for {pdf_path}: {e}", style="warning")
            finally:
                # Close doc in the finally block, ensuring it happens after try
                if temp_doc:
                    await asyncio.get_running_loop().run_in_executor(page_count_executor, temp_doc.close)  # Close moved here
        page_count_executor.shutdown(wait=True)
        console.print(f"Total pages to process: {total_pages_to_process}", style="info")

        phase2_pages_task_id = progress.add_task("[cyan]Processing Pages...", total=total_pages_to_process)
        phase2_start_time = time.monotonic()
        gpu_semaphore = asyncio.Semaphore(max_workers)
        gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='GPUWorker')

        processed_data_for_phase3 = []
        for pdf_path, project_dir, _, image_paths in successful_extractions:
            current_pdf_name = Utils.get_file_name(pdf_path)
            progress.update(phase2_docs_task_id, description=f"[bold magenta]Phase 2: Processing {current_pdf_name}...")

            success, _, pdf_name, generated_content = await _process_pdf_content(
                pdf_path, project_dir, image_paths, gpu_executor, gpu_semaphore, data, progress, phase2_pages_task_id
            )

            if success:
                processed_data_for_phase3.append((project_dir, pdf_name, generated_content))
            progress.update(phase2_docs_task_id, advance=1)

            del generated_content
            gc.collect()
            if TORCH and TORCH.cuda.is_available():
                TORCH.cuda.empty_cache()

        gpu_executor.shutdown(wait=True)
        progress.update(phase2_pages_task_id, completed=total_pages_to_process, description="[green]Page Processing Complete")
        progress.update(phase2_docs_task_id, description="[bold green]Phase 2: Document Processing Complete")
        # Calculate counts but don't print yet
        successful_processing_count = len(processed_data_for_phase3)
        failed_processing_count = successful_extractions_count - successful_processing_count

        if not processed_data_for_phase3:
            # Print immediate exit message if needed
            console.print("No documents successfully processed for content. Exiting.", style="warning")
            return  # Exit before printing summaries

        phase3_task_id = progress.add_task("[bold yellow]Phase 3: Assembling LaTeX Files...", total=len(processed_data_for_phase3))
        phase3_start_time = time.monotonic()
        for project_dir, pdf_name, content_list in processed_data_for_phase3:
            success = await _assemble_latex_file(project_dir, pdf_name, content_list, progress, phase3_task_id)
            if success:
                assembly_success_count += 1

        progress.update(phase3_task_id, completed=len(processed_data_for_phase3), description="[bold green]Phase 3: LaTeX Assembly Complete")
        # Calculate counts but don't print yet
        failed_assembly_count = len(processed_data_for_phase3) - assembly_success_count

    # --- End of `with Progress` block ---

    # Print summaries AFTER the progress bars are finished
    console.print(f"\n--- Summary ---", style="bold")
    console.print(f"Phase 1 (Image Extraction): {successful_extractions_count} succeeded, {failed_extractions_count} failed.", style="info")
    console.print(f"Phase 2 (Content Processing): {successful_processing_count} succeeded, {failed_processing_count} failed.", style="info")
    console.print(f"Phase 3 (LaTeX Assembly):   {assembly_success_count} succeeded, {failed_assembly_count} failed.", style="info")

    total_duration = time.monotonic() - start_time
    console.print(f"\nTotal Conversion Time: {total_duration:.2f}s", style="bold green")


def convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER, max_workers=1, image_workers=4):
    """Synchronously convert a PDF file or directory of PDF files to LaTeX."""
    try:
        asyncio.run(async_convert(source_path, output_dir, data, max_workers, image_workers))
    except Exception as e:
        print_error(f"An top-level error occurred during conversion: {e}")
        console.print(traceback.format_exc(), style="dim")
