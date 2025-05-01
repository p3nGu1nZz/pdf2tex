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
import importlib
import shutil

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
BATCH_SIZE = None

# Ensure TORCH is imported if not already
try:
    import torch as torch_module
except ImportError:
    TORCH = None  # Handle case where torch might not be installed initially

# --- Dependency Loading Function ---
def _ensure_dependencies_loaded(batch_size=16, quantize=True):
    """Loads heavy dependencies and initializes READER if not already done."""
    global READER, CV2, FITZ, PLT, NP, TORCH, EASYOCR, IS_LOADED, BATCH_SIZE

    # Store batch size globally so it can be accessed throughout the module
    BATCH_SIZE = batch_size
    
    # Skip if already loaded
    if IS_LOADED:
        return

    try:
        # Dynamically import heavy libraries
        cv2_module = importlib.import_module("cv2")
        fitz_module = importlib.import_module("fitz")
        plt_module = importlib.import_module("matplotlib.pyplot")
        np_module = importlib.import_module("numpy")
        torch_module = importlib.import_module("torch")
        easyocr_module = importlib.import_module("easyocr")

        CV2 = cv2_module
        FITZ = fitz_module
        PLT = plt_module
        NP = np_module
        TORCH = torch_module
        EASYOCR = easyocr_module

        # Set torch num_threads to match number of physical cores
        cpu_threads = max(2, os.cpu_count() // 2)  # A reasonable default
        TORCH.set_num_threads(cpu_threads)  # Increased from 1

        # Improve CUDA performance
        if TORCH.cuda.is_available():
            TORCH.cuda.set_device(0)
            TORCH.backends.cudnn.benchmark = True  # Enable benchmark mode

        console.print("Initializing EasyOCR Reader...", style="info")
        console.print(f"Using batch size: {batch_size}, quantize: {quantize}", style="info")
        
        # Use the quantize parameter
        READER = EASYOCR.Reader(['en'], gpu=TORCH.cuda.is_available(), quantize=quantize)

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


async def _process_pdf_content(pdf_path, project_dir, image_paths, gpu_executor, gpu_semaphore, data_folder,
                             progress, page_task_id, batch_size=16):  # Add batch_size parameter
    """
    Helper coroutine for Phase 2: Processes content.
    Returns tuple: (success_flag, project_dir, pdf_name, body_content_list, bibtex_string_list)
    """
    pdf_name = Utils.get_file_name(pdf_path)
    pdf = None
    body_content = []
    bibtex_entries = []
    try:
        await gpu_semaphore.acquire()
        try:
            pdf = await PDF.async_init(
                pdf_path, NP, CV2, FITZ, READER, gpu_executor, console, gpu_semaphore,
                extracted_image_paths=image_paths,
                data_folder=data_folder,
                output_dir=project_dir,
                progress=progress,
                page_task_id=page_task_id,
                batch_size=batch_size  # Pass batch_size to PDF
            )
            if not pdf:
                console.print(f"[Phase 2] Failed to initialize PDF object for {pdf_name}", style="danger")
                return False, project_dir, pdf_name, [], []

            body_content, bibtex_entries = await pdf.async_generate_latex_content()

            return True, project_dir, pdf_name, body_content, bibtex_entries
        finally:
            gpu_semaphore.release()

    except Exception as e:
        console.print(f"[Phase 2] Error processing content for {pdf_name}: {e}", style="danger")
        console.print(traceback.format_exc(), style="dim")
        if pdf and pdf.num_pages > 0 and progress and page_task_id is not None:
            try:
                pass
            except Exception:
                pass
        return False, project_dir, pdf_name, [], []


async def _assemble_latex_file(project_dir, pdf_name, body_content_list, bibtex_string_list, console_instance, progress, task_id):
    """
    Helper coroutine for Phase 3: Assembles main.tex, body.tex, and references.bib.
    """
    success = False
    lines_written_main = 0
    lines_written_body = 0
    lines_written_bib = 0

    main_tex_path = Utils.safe_join(project_dir, "main.tex")
    body_tex_path = Utils.safe_join(project_dir, "body.tex")
    bib_path = Utils.safe_join(project_dir, "references.bib")

    try:
        main_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}} % Required for including images
\\usepackage{{amsmath}} % For math environments
\\usepackage{{geometry}} % For page layout adjustments
\\usepackage[utf8]{{inputenc}} % Input encoding
\\usepackage[T1]{{fontenc}} % Font encoding
\\usepackage{{float}} % For [H] placement specifier
\\usepackage{{hyperref}} % For clickable links (optional)

\\graphicspath{{{{./assets/}}}} % Tell LaTeX where to find images

\\title{{{Utils.escape_special_chars(pdf_name)}}}
\\author{{Generated by PDF2Tex}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\input{{{os.path.basename(body_tex_path)}}} % Input the body content

\\clearpage % Ensure bibliography starts on a new page

\\bibliographystyle{{plain}} % Choose a bibliography style (e.g., plain, unsrt, alpha)
\\bibliography{{{os.path.splitext(os.path.basename(bib_path))[0]}}} % Reference the .bib file (without extension)

\\end{{document}}
"""
        body_string = "\n".join(map(str, body_content_list))

        from pdf2tex.references import ReferenceExtractor
        bib_string = ReferenceExtractor.merge_bibtex_entries(bibtex_string_list)

        write_tasks = [
            Utils.async_write_all(main_tex_path, main_content, console_instance),
            Utils.async_write_all(body_tex_path, body_string, console_instance),
            Utils.async_write_all(bib_path, bib_string, console_instance)
        ]
        results = await asyncio.gather(*write_tasks)
        lines_written_main, lines_written_body, lines_written_bib = results

        if lines_written_main > 0 and lines_written_body >= 0 and lines_written_bib >= 0:
            console_instance.print(f"Wrote {lines_written_main} lines to {main_tex_path}")
            console_instance.print(f"Wrote {lines_written_body} lines to {body_tex_path}")
            console_instance.print(f"Wrote {lines_written_bib} lines to {bib_path}")
            success = True
        else:
            console_instance.print(f"Failed to write one or more LaTeX files for {pdf_name}", style="danger")

    except Exception as e:
        console_instance.print(f"Error assembling LaTeX files for {pdf_name}: {e}", style="danger")
        console_instance.print(traceback.format_exc(), style="dim")
        success = False
    finally:
        if progress and task_id is not None:
            progress.update(task_id, advance=1)

    return success


async def async_convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER, max_workers=1, image_workers=4, 
                        batch_size=16, quantize=True):
    """Asynchronously convert PDF(s) using a three-phase approach with progress bars."""
    _ensure_dependencies_loaded(batch_size=batch_size, quantize=quantize)
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

    successful_extractions_count = 0
    failed_extractions_count = 0
    successful_processing_count = 0
    failed_processing_count = 0
    assembly_success_count = 0
    failed_assembly_count = 0
    total_pdfs_processed = len(pdf_paths_to_process)

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
        successful_extractions = [res for res in extraction_results if res[2]]
        successful_extractions_count = len(successful_extractions)
        failed_extractions_count = total_pdfs_processed - successful_extractions_count

        if not successful_extractions:
            console.print("No PDFs remaining after image extraction phase. Exiting.", style="warning")
            return

        phase2_docs_task_id = progress.add_task("[bold magenta]Phase 2: Processing Documents...", total=successful_extractions_count)
        total_pages_to_process = 0
        page_count_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='PageCount')
        temp_doc = None
        console.print("Calculating total pages for progress...", style="info")
        for pdf_path, _, _, _ in successful_extractions:
            temp_doc = None
            try:
                temp_doc = await asyncio.get_running_loop().run_in_executor(page_count_executor, FITZ.open, pdf_path)
                total_pages_to_process += temp_doc.page_count
            except Exception as e:
                console.print(f"Warning: Could not get page count for {pdf_path}: {e}", style="warning")
            finally:
                if temp_doc:
                    await asyncio.get_running_loop().run_in_executor(page_count_executor, temp_doc.close)
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

            success, _, pdf_name_ret, generated_body_content, generated_bib_entries = await _process_pdf_content(
                pdf_path, project_dir, image_paths, gpu_executor, gpu_semaphore, data, 
                progress, phase2_pages_task_id, batch_size  # Pass batch_size parameter
            )

            if success:
                processed_data_for_phase3.append((project_dir, pdf_name_ret, generated_body_content, generated_bib_entries))
            progress.update(phase2_docs_task_id, advance=1)

            del generated_body_content, generated_bib_entries
            gc.collect()
            if TORCH and TORCH.cuda.is_available():
                TORCH.cuda.empty_cache()

        gpu_executor.shutdown(wait=True)
        progress.update(phase2_pages_task_id, completed=total_pages_to_process, description="[green]Page Processing Complete")
        progress.update(phase2_docs_task_id, description="[bold green]Phase 2: Document Processing Complete")
        successful_processing_count = len(processed_data_for_phase3)
        failed_processing_count = successful_extractions_count - successful_processing_count

        if not processed_data_for_phase3:
            console.print("No documents successfully processed for content. Exiting.", style="warning")
            return

        phase3_task_id = progress.add_task("[bold yellow]Phase 3: Assembling LaTeX Files...", total=len(processed_data_for_phase3))
        phase3_start_time = time.monotonic()
        assembly_tasks = []
        for proj_dir, pdf_name_ph3, body_content_ph3, bib_entries_ph3 in processed_data_for_phase3:
            assembly_tasks.append(
                _assemble_latex_file(proj_dir, pdf_name_ph3, body_content_ph3, bib_entries_ph3, console, progress, phase3_task_id)
            )
        assembly_results = await asyncio.gather(*assembly_tasks)
        assembly_success_count = sum(assembly_results)

        progress.update(phase3_task_id, completed=len(processed_data_for_phase3), description="[bold green]Phase 3: LaTeX Assembly Complete")
        failed_assembly_count = len(processed_data_for_phase3) - assembly_success_count

    console.print(f"\n--- Summary ---", style="bold")
    console.print(f"Phase 1 (Image Extraction): {successful_extractions_count} succeeded, {failed_extractions_count} failed.", style="info")
    console.print(f"Phase 2 (Content Processing): {successful_processing_count} succeeded, {failed_processing_count} failed.", style="info")
    console.print(f"Phase 3 (LaTeX Assembly):   {assembly_success_count} succeeded, {failed_assembly_count} failed.", style="info")

    total_duration = time.monotonic() - start_time
    console.print(f"\nTotal Conversion Time: {total_duration:.2f}s", style="bold green")


def convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER, max_workers=1, image_workers=4, 
           batch_size=16, quantize=True):
    """Synchronously convert a PDF file or directory of PDF files to LaTeX."""
    try:
        asyncio.run(async_convert(source_path, output_dir, data, max_workers, image_workers, 
                                 batch_size, quantize))
    except Exception as e:
        print_error(f"An unexpected error occurred during conversion: {e}")
        console.print(traceback.format_exc(), style="dim")
        gc.collect()
        if TORCH and TORCH.cuda.is_available():
            TORCH.cuda.empty_cache()
        sys.exit(1)
