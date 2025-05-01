"""
PDF to LaTeX Converter Package

This package provides tools to convert PDF files to LaTeX format,
extracting text using OCR and handling non-textual elements as figures.
"""

import os
import sys
import click

__version__ = "0.1.2"

# Import main functions/classes needed by CLI or public API
from .main import (
    convert,
    async_convert,
    BBox,
    _ensure_dependencies_loaded
)

# Import utility modules
from .utils import Utils
from .command import Command
from .latex_text import LatexText
from .environment import Environment
from .constants import DEFAULT_DATA_FOLDER

# Import core classes
from .pdf import PDF
from .tex_file import TexFile
from .page import Page
from .block import Block

# Import UI elements
from .ui import console

# Define public API
__all__ = [
    "convert",
    "async_convert",
    "PDF",
    "TexFile",
    "Block",
    "Page",
    "LatexText",
    "Command",
    "Environment",
    "BBox",
    "Utils",
    "console",
    "main_cli",
]


def convert_pdf(pdf_path, output_dir=".", data="data"):
    """
    Convert a single PDF file to LaTeX format.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the output files.
        data (str): Directory for storing intermediate files.

    Returns:
        None
    """
    _ensure_dependencies_loaded()
    from .main import convert  # Local import to avoid circular dependencies
    return convert(pdf_path, output_dir, data)


def convert_pdfs_in_directory(directory_path, output_dir=".", data="data"):
    """
    Convert all PDF files in a directory to LaTeX format.

    Args:
        directory_path (str): Path to the directory containing PDF files.
        output_dir (str): Directory to save the output files.
        data (str): Directory for storing intermediate files.

    Returns:
        None
    """
    _ensure_dependencies_loaded()
    from .main import convert  # Local import to avoid circular dependencies
    return convert(directory_path, output_dir, data)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--file", "-f", type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the PDF file to convert."
)
@click.option(
    "--path", "-p", type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to the directory containing PDFs to convert."
)
@click.option(
    "--output", "-o", type=click.Path(resolve_path=True), default=".", show_default=True,
    help="Output directory for generated LaTeX projects."
)
@click.option(
    "--data", "-d", type=click.Path(resolve_path=True), default=DEFAULT_DATA_FOLDER, show_default=True,
    help="Directory for storing intermediate files (build artifacts)."
)
@click.option(
    "--max-workers", "-w", type=int, default=1, show_default=True,  # Added -w shortcut
    help="Max workers for OCR/content processing (Phase 2)."
)
@click.option(
    "--image-workers", "-i", type=int, default=4, show_default=True,  # Added -i shortcut
    help="Max workers for image extraction (Phase 1)."
)
@click.option(
    "--batch-size", "-b", type=int, default=16, show_default=True,  # Added -b shortcut
    help="Batch size for OCR operations (higher values use more VRAM but may be faster)."
)
@click.option(
    "--quantize/--no-quantize",  # Use a flag with automatic boolean handling
    default=True,  # Default is to quantize
    show_default=True,
    help="Use quantized models (--quantize, default) or full precision (--no-quantize)."
)
@click.version_option(version=__version__, message="PDF2Tex %(version)s")
def main_cli(file, path, output, data, max_workers, image_workers, batch_size, quantize):  # Add new parameters
    """
    PDF2TEX - Convert PDF files to LaTeX format.

    This tool extracts text using OCR (EasyOCR) and identifies/extracts images
    from PDF documents, generating structured LaTeX projects.

    It processes files asynchronously in three phases:
    1. Image Extraction (using --image-workers)
    2. Content Processing (OCR, using --max-workers)
    3. LaTeX File Assembly

    Specify either a single --file or a --path containing multiple PDFs.
    """
    # Load dependencies only if not asking for help/version
    if not any(arg in sys.argv for arg in ["--help", "-h", "--version"]):
        # Pass batch_size and quantize to dependency loader
        _ensure_dependencies_loaded(batch_size=batch_size, quantize=quantize)

    if not file and not path:
        console.print("Error: Please provide either a --file or a --path option.", style="danger")
        ctx = click.get_current_context()
        console.print(ctx.get_help())
        sys.exit(1)

    # Ensure data directory exists
    os.makedirs(data, exist_ok=True)

    # Determine actual output directory
    project_output_dir = output
    os.makedirs(project_output_dir, exist_ok=True)

    console.print(f"Using output directory: {project_output_dir}", style="info")

    source_to_process = file if file else path

    # Call the main conversion function with all arguments, including new ones
    convert(source_to_process, project_output_dir, data, max_workers, image_workers, batch_size, quantize)


if __name__ == "__main__":
    main_cli()  # pylint: disable=no-value-for-parameter
