# PDF2LaTeX

Effortlessly convert multiple (or single) PDFs to LaTeX with a robust tool that extracts both images
and text. Integrate it into your project easily, or use on the command-line. CUDA compatible!

**Key Features:**

| Feature                     | Description                                                      |
| :-------------------------- | :--------------------------------------------------------------- |
| 📄 PDF to LaTeX             | Converts PDF documents into structured LaTeX (`.tex`) files.     |
| 🔍 OCR Text Extraction      | Uses EasyOCR to accurately extract text content, even from       |
| 🖼️ Image Handling           | Identifies and extracts figures/images, saving them separately.  |
| ⚡ Asynchronous Processing   | Leverages `asyncio` for significantly faster parallel processing.|
| 🚀 GPU Acceleration         | Supports CUDA for GPU-accelerated OCR (optional setup).          |
| 💻 CLI & API   | Can be used as a command-line tool or imported into Python script             |

## Setup

Easily install `pdf2tex` using the pypi package

```bash
pip install pdf2tex
```

> `venv` support is optional with `python -m venv .venv`

### CUDA Support

To enable CUDA support for GPU acceleration use the following command

```bash
# remove CPU version of torch
pip uninstall torch torchvision

# install CUDA from wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Command-Line Usage

```bash
# Convert a single PDF
pdf2tex --file path/to/your/document.pdf --output ./output_project

# Convert all PDFs in a directory
pdf2tex --path path/to/your/documents/ --output ./output_projects --data-dir ./temp_data
```

**Options:**

* `--file, -f`: Path to the PDF file to convert.
* `--path, -p`: Path to the directory containing PDFs to convert.
* `--output, -o`: Output directory for generated LaTeX projects (default: current directory).
* `--data-dir, -d`: Directory for storing intermediate files (default: `data`).
* `--version`: Show the version and exit.
* `--help, -h`: Show help message and exit.

## Programmatic Usage

You can also use `pdf2tex` directly within your Python scripts:

### Converting a Single PDF

```python
import pdf2tex

# Basic usage with defaults (output to current directory, 'data' for intermediate files)
pdf2tex.convert_pdf('path/to/document.pdf')

# Specify output and data directories
pdf2tex.convert_pdf(
    pdf_path='path/to/document.pdf',
    output_dir='./my_output_folder',
    data_dir='./my_temp_data'
)
```

### Converting Multiple PDFs

```python
import pdf2tex

# Convert all PDF files in a directory
pdf2tex.convert_pdfs_in_directory(
    directory_path='path/to/pdf_folder',
    output_dir='./latex_projects_output',
    data_dir='./conversion_temp'
)
```

### API Reference

The main functions available for programmatic use are:

* `convert_pdf(pdf_path, output_dir='.', data_dir='data')`: Convert a single PDF file.
* `convert_pdfs_in_directory(directory_path, output_dir='.', data_dir='data')`: Convert all PDFs in a directory.
* `convert(source_path, output_dir='.', data_dir=DEFAULT_DATA_FOLDER)`: Core function that handles both files and directories.
* `async_convert(source_path, output_dir='.', data_dir=DEFAULT_DATA_FOLDER)`: Asynchronous version of convert.

All parameters match the CLI arguments:
* `pdf_path`/`source_path`: Path to the PDF file to convert.
* `directory_path`/`source_path`: Path to the directory containing PDFs to convert.
* `output_dir`: Output directory for generated LaTeX projects (default: current directory).
* `data_dir`: Directory for storing intermediate files (default: `data`).

## Development

To begin developing your own features or contribution to `pdf2tex` follow the instructions below. 

```bash
# Clone the repo using git
git clone https://github.com/p3nGu1nZz/pdf2tex.git
cd pdf2tex

# Create environment
python -m venv .venv

# Activate - Linux
.venv\Source\activate  

# or

# Activate - Windows
.venv\Scripts\activate

# Install editable package
pip install -e .[dev]

```

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

## Citations

Please use the following BibTeX entry to cite this project:

```bibtex
@software{oarc,
    author = {Kara Rawson},
    title = {PDF2LaTeX: Effortlessly convert PDFs to LaTeX with image extraction.},
    date = {4-27-2025},
    howpublished = {\url{https://github.com/p3nGu1nZz/pdf2tex}}
}
```

## Contact

For questions or support, please contact us at:

- **Email**: <backrqqms@gmail.com>
- **Issues**: [GitHub Issues](https://github.com/p3nGu1nZz/pdf2tex/issues)

