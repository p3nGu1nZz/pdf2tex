"""
Comprehensive tests for the pdf2tex module.
Tests core functionality, utility methods, and conversion processes.
"""
# pylint: disable=redefined-outer-name, unused-argument

import os
import shutil
import sys
import tempfile
import unittest.mock as mock

import numpy as np
import pytest
import pytest_asyncio

import pdf2tex.pdf2tex as pdf2tex_module
from pdf2tex import convert_pdf, convert
from pdf2tex.pdf2tex import (
    BBox,
    Command,
    Environment,
    LatexText,
    Page,
    PDF,
    TexFile,
    Utils,
    _ensure_dependencies_loaded,
    async_convert,
    process_bboxes,
    safe_join,
)


# --- Test Fixtures ---
@pytest.fixture
def test_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pdf_path():
    """Return a mock PDF path."""
    return "sample.pdf"


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_pdf():
    """Create a mock PDF object."""
    pdf = mock.Mock(spec=PDF)
    pdf.name = "test_pdf"
    pdf.figures_dir = "figures"
    pdf.num_figs = 0
    return pdf


@pytest.fixture
def mock_page(mock_pdf, mock_image):
    """Create a mock Page object."""
    page = mock.Mock(spec=Page)
    page.parent_pdf = mock_pdf
    page.page_img = mock_image
    page.height = 100
    page.width = 100
    return page


@pytest.fixture
def mock_bbox():
    """Create a mock BBox object."""
    return BBox(10, 20, 30, 40)


@pytest.fixture
def loaded_dependencies():
    """Ensure dependencies are loaded for tests."""
    with mock.patch('pdf2tex.pdf2tex._ensure_dependencies_loaded') as mock_load:
        # Mock successful dependency loading using the imported module
        pdf2tex_module.IS_LOADED = True
        pdf2tex_module.CV2 = mock.MagicMock()
        pdf2tex_module.NP = np
        pdf2tex_module.FITZ = mock.MagicMock()
        pdf2tex_module.READER = mock.MagicMock()
        yield mock_load


# --- Unit Tests ---
class TestUtils:
    """Tests for the Utils class."""

    def test_get_file_name(self):
        """Test extracting filename without extension."""
        assert Utils.get_file_name("/path/to/file.pdf") == "file"
        assert Utils.get_file_name("file.name.with.dots.pdf") == "file.name.with.dots"
        assert Utils.get_file_name("file") == "file"
        assert Utils.get_file_name("file.") == "file"

    def test_escape_special_chars(self):
        """Test escaping LaTeX special characters."""
        input_str = r"Test & % $ # _ { } ~ ^ \\"
        # Each backslash is escaped as \textbackslash{}, so two backslashes yield two escapes
        expected = r"Test \& \% \$ \# \_ \{ \} \textasciitilde{} \textasciicircum{} \textbackslash{}\textbackslash{}"
        assert Utils.escape_special_chars(input_str) == expected

    def test_escape_special_chars_backslash(self):
        """Test escaping LaTeX special characters with multiple backslashes."""
        input_str = r"Test & % $ # _ { } ~ ^ \\"
        # Each backslash is escaped as \textbackslash{}, so two backslashes yield two escapes
        expected = r"Test \& \% \$ \# \_ \{ \} \textasciitilde{} \textasciicircum{} \textbackslash{}\textbackslash{}"
        assert Utils.escape_special_chars(input_str) == expected

    def test_make_strlist(self):
        """Test converting list items to strings."""
        input_list = [1, 2.5, "three", None]
        expected = ["1", "2.5", "three", "None"]
        assert Utils.make_strlist(input_list) == expected

    def test_create_latex_project_structure(self, test_directory):
        """Test creating LaTeX project structure."""
        # Pass the temporary directory directly as the base path
        result = Utils.create_latex_project_structure(test_directory, "test_pdf")
        assert "project_dir" in result
        assert "figures_dir" in result
        assert "build_dir" in result
        # Check paths relative to the test_directory
        assert result["project_dir"] == os.path.join(test_directory, "test_pdf")
        assert os.path.exists(result["project_dir"])
        assert os.path.exists(result["figures_dir"])
        assert os.path.exists(result["build_dir"])

    def test_pct_white(self):
        """Test calculating percentage of white pixels."""
        # Create an image with 50% white pixels
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[0:5, :, :] = 255  # Make half the image white
        # Ensure CV2 is mocked if dependencies aren't fully loaded
        with mock.patch('pdf2tex.pdf2tex.CV2', mock.MagicMock()) as mock_cv2:
            mock_cv2.split.return_value = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
            # Ensure NP is available or mocked
            with mock.patch('pdf2tex.pdf2tex.NP', np):
                assert abs(Utils.pct_white(img) - 0.5) < 0.01

    def test_remove_duplicate_bboxes(self):
        """Test removing duplicate bounding boxes."""
        bbox1 = BBox(0, 10, 100, 20)
        bbox2 = BBox(0, 10, 200, 30)  # Same y as bbox1
        bbox3 = BBox(0, 20, 100, 20)

        result = Utils.remove_duplicate_bboxes([bbox1, bbox2, bbox3])
        assert len(result) == 2
        assert result[0].y == 10
        assert result[1].y == 20


class TestBBox:
    """Tests for the BBox class."""

    def test_init(self):
        """Test initialization of BBox."""
        bbox = BBox(10, 20, 30, 40)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 30
        assert bbox.height == 40
        assert bbox.y_bottom == 60  # y + height


class TestLatexClasses:
    """Tests for LaTeX-related classes."""

    def test_latex_text(self):
        """Test LatexText class."""
        text = LatexText("Test & text")
        # Expected output after fixing the escape logic
        assert text.text == r"Test \& text"

    def test_command(self):
        """Test Command class."""
        # Simple command
        cmd = Command("textbf", arguments=["bold text"])
        assert cmd.text == r"\textbf{bold text}"

        # Command with options
        cmd = Command("includegraphics",
                      arguments=["image.png"],
                      options=[("width", r"0.8\textwidth")])
        assert cmd.text == r"\includegraphics[width=0.8\textwidth]{image.png}"

    def test_environment(self):
        """Test Environment class."""
        content = [LatexText("Content text")]
        env = Environment(content, "center")
        assert len(env.content) == 3  # begin + content + end
        assert env.content[0].text == r"\begin{center}"
        assert env.content[2].text == r"\end{center}"


class TestSegmentation:
    """Tests for PDF segmentation functions."""

    def test_process_bboxes(self):
        """Test processing bounding boxes."""
        bbox1 = BBox(0, 10, 100, 30)  # y_bottom = 40
        bbox2 = BBox(0, 35, 100, 30)  # Overlaps with bbox1

        with mock.patch('pdf2tex.pdf2tex.Utils.remove_duplicate_bboxes',
                        return_value=[bbox1, bbox2]):
            with mock.patch('pdf2tex.pdf2tex.Utils.merge_bboxes',
                            return_value=[bbox1, bbox2]):
                result = process_bboxes([bbox1, bbox2])

                # Check if y_bottom of first box and y of second were adjusted
                assert result[0].y_bottom == 37  # (40+35)/2 = 37.5 -> 37
                assert result[1].y == 37


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Tests for asynchronous functions."""

    async def test_async_write_all(self, test_directory):
        """Test asynchronous file writing."""
        test_file = os.path.join(test_directory, "test.txt")
        content = ["Line 1", "Line 2", "Line 3"]

        result = await Utils.async_write_all(test_file, content)
        assert result is True

        # Check file content
        with open(test_file, 'r', encoding='utf-8') as f:
            file_content = f.read().splitlines()
        assert file_content == content

    @pytest_asyncio.fixture
    async def mock_async_pdf(self):
        """Create a mock async PDF object."""
        pdf = mock.AsyncMock(spec=PDF)
        pdf.name = "test_pdf"
        pdf.project_dir = "project_dir"
        pdf.figures_dir = "figures_dir"
        pdf.async_generate_latex.return_value = [
            Environment([LatexText("Test content")], "document")
        ]
        return pdf

    async def test_texfile_async_init(self, mock_async_pdf):
        """Test asynchronous initialization of TexFile."""
        tex_file = await TexFile.async_init(mock_async_pdf)
        assert tex_file.pdf_obj == mock_async_pdf
        assert len(tex_file.preamble) > 0
        assert len(tex_file.body) > 0

    async def test_pdf_async_init(self, sample_pdf_path, test_directory):
        """Test asynchronous initialization of PDF."""
        # Use AsyncMock for extract_images_from_pdf to return the list directly on await
        with mock.patch('pdf2tex.pdf2tex.Utils.extract_images_from_pdf',
                        new_callable=mock.AsyncMock,
                        return_value=["image1.png", "image2.png"]) as mock_extract:

            # Mock _async_pdf_to_pages as before
            with mock.patch('pdf2tex.pdf2tex.PDF._async_pdf_to_pages',
                            new_callable=mock.AsyncMock,
                            return_value=[]) as mock_pages:

                # Pass test_directory as output_dir to avoid safe_join issues
                pdf = await PDF.async_init(sample_pdf_path, test_directory, test_directory)
                assert pdf.path == sample_pdf_path
                # data_folder is passed as test_directory
                assert pdf.data_folder == test_directory
                # project_dir should be inside test_directory
                assert pdf.project_dir.startswith(test_directory)
                assert len(pdf.embedded_images) == 2
                # Assert mocks were awaited
                mock_extract.assert_awaited_once()
                mock_pages.assert_awaited_once()


class TestConversion:
    """Tests for conversion functions."""

    @pytest.mark.asyncio
    async def test_async_convert_single_file(self, loaded_dependencies, test_directory):
        """Test asynchronous conversion of a single file."""
        output_dir = test_directory
        data_dir = test_directory
        with mock.patch('pdf2tex.pdf2tex.PDF.async_init') as mock_pdf_init:
            mock_pdf = mock.AsyncMock()
            mock_pdf.project_dir = "project_dir"
            mock_pdf.name = "test_pdf"
            mock_pdf_init.return_value = mock_pdf

            with mock.patch('pdf2tex.pdf2tex.TexFile.async_init') as mock_tex_init:
                mock_tex = mock.AsyncMock()
                mock_tex_init.return_value = mock_tex

                with mock.patch('pdf2tex.pdf2tex.Utils.get_file_name',
                                return_value="test_pdf"):
                    output_tex_path = os.path.join(output_dir, "test_pdf", "test_pdf.tex")
                    with mock.patch('pdf2tex.pdf2tex.safe_join',
                                    return_value=output_tex_path):
                        with mock.patch('builtins.open', mock.mock_open()):
                            await async_convert("test.pdf", output_dir, data_dir)
                            mock_pdf_init.assert_called_once()
                            mock_tex_init.assert_called_once()
                            mock_tex.async_generate_tex_file.assert_called_once()

    def test_convert(self, loaded_dependencies, test_directory):
        """Test synchronous conversion function."""
        output_dir = test_directory
        data_dir = test_directory
        with mock.patch('asyncio.run') as mock_run:
            convert("test.pdf", output_dir, data_dir)
            mock_run.assert_called_once()

    def test_convert_pdf(self, test_directory):
        """Test convert_pdf convenience function."""
        output_dir = test_directory
        data_dir = test_directory

        # Patch convert in the pdf2tex package namespace (where convert_pdf will look for it)
        with mock.patch('pdf2tex.convert') as mock_convert:
            # Mock safe_join to handle any path safety checks
            with mock.patch('pdf2tex.pdf2tex.safe_join', return_value=os.path.join(output_dir, "test")):
                # Create a test file to prevent file not found errors
                test_file = os.path.join(test_directory, "test.pdf")
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write("dummy content")

                # Call the function imported from pdf2tex (package)
                convert_pdf(test_file, output_dir, data_dir)

                # Verify the convert function was called with correct args
                mock_convert.assert_called_once_with(test_file, output_dir, data_dir)


class TestSafety:
    """Tests for safety functions."""

    def test_safe_join(self):
        """Test safe path joining."""
        base = os.path.abspath("/base/path")

        # Valid path
        assert safe_join(base, "subdir") == os.path.join(base, "subdir")

        # Path traversal attempt
        with pytest.raises(ValueError):
            safe_join(base, "../../../etc/passwd")


class TestDependencyLoading:
    """Tests for dependency loading."""

    @pytest.mark.filterwarnings("ignore:coroutine '.*' was never awaited")
    def test_ensure_dependencies_loaded(self):
        """Test dependency loading function by patching sys.modules."""
        pdf2tex_module.IS_LOADED = False

        # Create mocks for all required modules
        mock_cv2 = mock.MagicMock(spec=['split', 'cvtColor', 'adaptiveThreshold', 'getStructuringElement', 'morphologyEx', 'dilate', 'findContours', 'boundingRect', 'addWeighted', 'imwrite', 'COLOR_BGR2GRAY', 'ADAPTIVE_THRESH_MEAN_C', 'THRESH_BINARY_INV', 'MORPH_RECT', 'MORPH_GRADIENT', 'MORPH_CLOSE', 'RETR_EXTERNAL', 'CHAIN_APPROX_SIMPLE', 'COLOR_RGBA2BGR', 'COLOR_RGB2BGR', 'COLOR_GRAY2BGR'])
        mock_fitz = mock.MagicMock(spec=['open'])
        mock_plt = mock.MagicMock(spec=['imshow', 'show'])
        mock_np = mock.MagicMock(spec=np)  # Use real numpy spec for better mocking
        mock_torch = mock.MagicMock(spec=['cuda'])
        mock_torch.cuda.is_available.return_value = False
        mock_easyocr_module = mock.MagicMock(spec=['Reader'])
        mock_reader_instance = mock.MagicMock()
        mock_easyocr_module.Reader.return_value = mock_reader_instance

        # Modules dictionary for patching sys.modules
        mock_modules = {
            'cv2': mock_cv2,
            'fitz': mock_fitz,
            'matplotlib.pyplot': mock_plt,
            'numpy': mock_np,
            'torch': mock_torch,
            'easyocr': mock_easyocr_module,
            # Include mocks for matplotlib's internal checks if needed, though often covered by mocking plt
            'cycler': mock.MagicMock(__version__='1.0.0'),
            'dateutil': mock.MagicMock(__version__='2.8.0'),
            'kiwisolver': mock.MagicMock(__version__='1.3.2'),
            'pyparsing': mock.MagicMock(__version__='3.0.0'),
            'matplotlib': mock.MagicMock(__version__='3.0.0', pyplot=mock_plt)  # Mock base matplotlib too
        }

        # Patch console output and sys.exit
        with mock.patch('pdf2tex.pdf2tex.console'), \
             mock.patch('sys.exit') as mock_sys_exit:
            # Patch sys.modules to make the mocks available for import
            with mock.patch.dict(sys.modules, mock_modules):
                # Run the function - it should now import the mocks
                _ensure_dependencies_loaded()

                # Assertions
                mock_sys_exit.assert_not_called()  # Ensure no import error occurred
                mock_easyocr_module.Reader.assert_called_once()  # Check if Reader was initialized
                assert pdf2tex_module.IS_LOADED is True  # Check loaded flag
                assert pdf2tex_module.READER is mock_reader_instance  # Check global assignment

                # Reset call count for the second call check
                mock_easyocr_module.Reader.reset_mock()

                # Second call should not re-initialize
                _ensure_dependencies_loaded()
                mock_easyocr_module.Reader.assert_not_called()  # Reader should not be called again


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
