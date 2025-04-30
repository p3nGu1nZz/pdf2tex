"""
PDF to LaTeX Converter Package

This package provides tools to convert PDF files to LaTeX format,
extracting text using OCR and handling non-textual elements as figures.
"""

import os
import sys
import click

__version__ = "1.0.0"

# Import main functions/classes from the module where the code exists
from .pdf2tex import (  # noqa
    convert,
    async_convert,
    PDF,
    TexFile,
    Block,
    Page,
    LatexText,
    Command,
    Environment,
    BBox,
    Utils,
    safe_join,
    _ensure_dependencies_loaded,
    DEFAULT_DATA_FOLDER,
    console
)

# Define what is available when doing 'from pdf2tex import *'
# Also helps tools understand the public API
__all__ = [
    'convert',
    'async_convert',
    'PDF',
    'TexFile',
    'Block',
    'Page',
    'LatexText',
    'Command',
    'Environment',
    'BBox',
    'Utils',
    'safe_join',
    'convert_pdf',
    'convert_pdfs_in_directory',
    'main'
]


# Helper functions for convenience
def convert_pdf(pdf_path, output_dir='.', data_dir='data'):
    """
    Convert a PDF file to LaTeX format.

    Args:
        pdf_path (str): Path to the PDF file to convert
        output_dir (str): Directory to store output files
        data_dir (str): Directory to store intermediate files

    Returns:
        None
    """
    # Make sure dependencies are loaded before calling convert
    _ensure_dependencies_loaded()
    return convert(pdf_path, output_dir, data_dir)


def convert_pdfs_in_directory(directory_path, output_dir='.', data_dir='data'):
    """
    Convert all PDF files in a directory to LaTeX format.

    Args:
        directory_path (str): Path to directory containing PDF files
        output_dir (str): Directory to store output files
        data_dir (str): Directory to store intermediate files

    Returns:
        None
    """
    # Make sure dependencies are loaded before calling convert
    _ensure_dependencies_loaded()
    return convert(directory_path, output_dir, data_dir)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--file', '-f', type=click.Path(exists=True),
    help="Path to PDF file to convert"
)
@click.option(
    '--path', '-p', type=click.Path(exists=True),
    help="Path to directory containing PDFs to convert"
)
@click.option(
    '--output', '-o', type=click.Path(), default='.', show_default=True,
    help="Output directory for generated LaTeX projects"
)
@click.option(
    '--data-dir', '-d', type=click.Path(), default=DEFAULT_DATA_FOLDER,
    show_default=True, help="Directory for storing intermediate files"
)
@click.version_option(version=__version__)
def main(file, path, output, data_dir):
    """
    PDF2TEX - Convert PDF files to LaTeX format.

    This tool extracts text using OCR and treats non-textual elements as figures.
    It processes files in parallel using asyncio for improved performance.

    Each PDF is converted to a complete LaTeX project with proper directory
    structure.

    Examples:

        # Convert a single PDF file:
        pdf2tex --file document.pdf --output ./projects

        # Convert all PDF files in a directory:
        pdf2tex --path ./documents --output ./projects
    """
    # --- Deferred Imports ---
    # Only load dependencies for actual execution (not --help)
    # Skip loading here if --help is in sys.argv
    if '--help' not in sys.argv and '-h' not in sys.argv:
        _ensure_dependencies_loaded()
    # --- End Deferred Imports ---

    if not file and not path:
        console.print(
            "Error: Please provide either --file or --path option.",
            style="danger"
        )
        console.print("Use --help to see usage information.", style="info")
        sys.exit(1)

    # Ensure data_dir exists first, as it might be used in the output path
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    # Determine the final output directory for projects
    project_output_dir = output
    if output == '.':
        # If default output is used, place projects inside data_dir
        # relative to the current directory.
        project_output_dir = safe_join(os.getcwd(), data_dir)
        console.print(
            f"No output directory specified. Using data directory: "
            f"{project_output_dir}",
            style="info"
        )
    else:
        # If a specific output dir is given, ensure it exists
        os.makedirs(project_output_dir, exist_ok=True)
        console.print(
            f"Using specified output directory: {project_output_dir}",
            style="info"
        )

    # Pass the potentially modified project_output_dir and original data_dir
    if path:
        convert(path, project_output_dir, data_dir)
    elif file:
        convert(file, project_output_dir, data_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
