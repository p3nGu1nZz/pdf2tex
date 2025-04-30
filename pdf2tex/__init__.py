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

# Define public API
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


# Helper functions
def convert_pdf(pdf_path, output_dir='.', data='data'):
    """
    Convert a PDF file to LaTeX format.
    """
    _ensure_dependencies_loaded()
    return convert(pdf_path, output_dir, data)


def convert_pdfs_in_directory(directory_path, output_dir='.', data='data'):
    """
    Convert all PDF files in a directory to LaTeX format.
    """
    _ensure_dependencies_loaded()
    return convert(directory_path, output_dir, data)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--file', '-f', type=click.Path(exists=True), help="Path to PDF file to convert")
@click.option('--path', '-p', type=click.Path(exists=True), help="Path to directory containing PDFs to convert")
@click.option('--output', '-o', type=click.Path(), default='.', show_default=True, help="Output directory for generated LaTeX projects")
@click.option('--data', '-d', type=click.Path(), default=DEFAULT_DATA_FOLDER, show_default=True, help="Directory for storing intermediate files")
@click.version_option(version=__version__, message='PDF2Tex %(version)s')
def main(file, path, output, data):
    """
    PDF2TEX - Convert PDF files to LaTeX format.

    This tool extracts text using OCR and treats non-textual elements as figures.
    It processes files in parallel using asyncio for improved performance.
    """
    if '--help' not in sys.argv and '-h' not in sys.argv:
        _ensure_dependencies_loaded()

    if not file and not path:
        console.print("Error: Please provide either --file or --path option.", style="danger")
        console.print("Use --help to see usage information.", style="info")
        sys.exit(1)

    if data:
        os.makedirs(data, exist_ok=True)

    project_output_dir = output if output != '.' else safe_join(os.getcwd(), data)
    os.makedirs(project_output_dir, exist_ok=True)

    console.print(f"Using output directory: {project_output_dir}", style="info")

    if path:
        convert(path, project_output_dir, data)
    elif file:
        convert(file, project_output_dir, data)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
