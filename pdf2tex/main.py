# pylint: disable=too-many-lines
# pylint: disable=no-member
# pylint: disable=import-outside-toplevel, global-statement
"""
PDF to LaTeX Converter

This script converts PDF files to LaTeX format by extracting text using OCR
and handling non-textual elements as figures.
Uses asyncio and ThreadPoolExecutor for parallel processing.
"""

import os
import sys
import asyncio
import concurrent.futures  # Import concurrent.futures

from .utils import Utils
from .bbox import BBox
from .tex_file import TexFile
from .pdf import PDF
from .ui import console, progress_columns, custom_theme
# Import constants
from .constants import (
    DEFAULT_DATA_FOLDER,
    MIN_TEXT_SIZE,
    HORIZONTAL_POOLING,
    MAX_WORKERS
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


async def async_convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER):
    """Asynchronously convert a PDF file or directory of PDF files to LaTeX."""
    _ensure_dependencies_loaded()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

    if os.path.isdir(source_path):
        pdf_files = [f for f in os.listdir(source_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            console.print(f"No PDF files found in directory: {source_path}", style="warning")
            return

        tasks = []
        for filename in pdf_files:
            file_path = os.path.join(source_path, filename)
            file_output_dir_name = Utils.get_file_name(filename)
            file_output_dir = Utils.safe_join(output_dir, file_output_dir_name)
            tasks.append(async_convert(file_path, file_output_dir, data))  # Pass console if needed for recursive calls

        await asyncio.gather(*tasks)

    elif os.path.isfile(source_path) and source_path.lower().endswith('.pdf'):
        console.print(f"Processing file: {source_path}", style="info")
        console.print(f"Using output directory: {output_dir}", style="info")
        pdf = None  # Initialize pdf to None
        try:
            # Pass console instance to PDF.async_init
            pdf = await PDF.async_init(
                source_path, NP, CV2, FITZ, READER, executor, console, data, output_dir
            )
            if not pdf:
                console.print(f"Failed to initialize PDF for {source_path}", style="danger")
                executor.shutdown(wait=False)  # Ensure executor shutdown on early exit
                return

            try:
                tex_filename = Utils.safe_join(pdf.project_dir, f"{pdf.name}.tex")
                tex_file = await TexFile.async_init(pdf)
                await tex_file.async_generate_tex_file(tex_filename)
                console.print(
                    f"Successfully generated LaTeX project at\n{pdf.project_dir}",
                    style="success"
                )
            except (TypeError, ValueError, AttributeError) as tex_error:
                console.print(f"Failed to generate LaTeX for {source_path}: {tex_error}", style="danger")
                if pdf:  # Check if pdf was initialized
                    try:
                        console.print("Attempting to save partial results...", style="warning")
                        tex_filename = Utils.safe_join(pdf.project_dir, f"{pdf.name}_partial.tex")
                        basic_tex_file = TexFile(pdf)
                        basic_tex_file.generate_tex_file(tex_filename)
                        console.print(f"Saved partial results to {tex_filename}", style="info")
                    except (IOError, OSError) as partial_save_error:
                        console.print(f"Could not save partial results: {partial_save_error}", style="danger")
                    except Exception as partial_save_error:  # pylint: disable=broad-except
                        console.print(f"Unexpected error saving partial results: {partial_save_error}", style="danger")
                else:
                    console.print("Cannot save partial results as PDF object was not created.", style="warning")
            except Exception as tex_error:  # pylint: disable=broad-except
                console.print(f"Unexpected error generating LaTeX for {source_path}: {tex_error}", style="danger")
                if pdf:  # Check if pdf was initialized
                    try:
                        console.print("Attempting to save partial results...", style="warning")
                        tex_filename = Utils.safe_join(pdf.project_dir, f"{pdf.name}_partial.tex")
                        basic_tex_file = TexFile(pdf)
                        basic_tex_file.generate_tex_file(tex_filename)
                        console.print(f"Saved partial results to {tex_filename}", style="info")
                    except (IOError, OSError) as partial_save_error:
                        console.print(f"Could not save partial results: {partial_save_error}", style="danger")
                    except Exception as partial_save_error:  # pylint: disable=broad-except
                        console.print(f"Unexpected error saving partial results: {partial_save_error}", style="danger")
                else:
                    console.print("Cannot save partial results as PDF object was not created.", style="warning")
        except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as e:
            console.print(f"Error processing {source_path}: {str(e)}", style="danger")
            if "Block" in str(e) and "attribute" in str(e):
                console.print("The PDF structure could not be properly analyzed. Try a different PDF file.", style="danger")
        except Exception as e:  # pylint: disable=broad-except
            console.print(f"Unexpected critical error processing {source_path}: {e}", style="danger")
            import traceback
            traceback.print_exc()
        finally:  # Ensure executor is always shut down
            executor.shutdown(wait=True)
    else:
        print_error(f"Invalid source path: {source_path}")


def convert(source_path, output_dir='.', data=DEFAULT_DATA_FOLDER):
    """Synchronously convert a PDF file or directory of PDF files to LaTeX."""
    pdf = None  # Initialize pdf to None outside try block
    try:
        # Note: async_convert now handles executor creation/shutdown
        asyncio.run(async_convert(source_path, output_dir, data))
    except Exception as e:
        print_error(f"An error occurred during conversion: {e}")
        # Check if 'pdf' might exist in the scope where the error happened
        # This is tricky because the error might be deep inside asyncio.run
        # A more robust approach might involve async_convert returning status/partial data
        # For now, we add a basic check, but it might not always catch the pdf object
        pdf_obj_in_scope = None
        # This is a heuristic and might not work reliably depending on where the exception occurs
        # A better approach would be structured error handling returning partial state.
        # For simplicity, we'll rely on the checks within async_convert's except blocks.
        # print_status("Attempting to save partial results if possible...")
        # (Logic removed as it's unreliable here and handled better in async_convert)
