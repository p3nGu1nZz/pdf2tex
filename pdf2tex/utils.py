"""
Utility functions and classes for the PDF to LaTeX conversion process.
These include safe path handling, image processing, and LaTeX project structure creation.
"""

import os
import re
import asyncio

from .ui import console
from .bbox import BBox
from .constants import MIN_TEXT_SIZE, HORIZONTAL_POOLING


# --- Utility Classes and Functions ---
class Utils:
    """Static utility methods for various operations."""

    @staticmethod
    def safe_join(base, *paths):
        """
        Safely join one or more path components to the base directory.
        Prevents path traversal attacks by resolving and checking the final path.
        """
        base = os.path.abspath(base)
        final_path = os.path.abspath(os.path.join(base, *paths))
        if not final_path.startswith(base):
            raise ValueError(
                "Unsafe path detected (possible path traversal): " + final_path
            )
        return final_path

    @staticmethod
    def save_pil_images(items, path):
        """Save PIL Image items to folder specified by path."""
        safe_path = Utils.safe_join(os.getcwd(), path)
        if not os.path.isdir(safe_path):
            os.makedirs(safe_path, exist_ok=True)
        for idx, item in enumerate(items):
            save_path = Utils.safe_join(safe_path, f"{idx}.jpg")
            item.save(save_path)

    @staticmethod
    def pct_white(img, np_module, cv2_module, console):
        """Find percentage of white pixels in img."""
        if np_module is None:
            console.print("Error: NumPy not loaded.", style="danger")
            return 0
        if img is None or img.size == 0:
            return 1.0  # Treat empty/None image as fully white

        white_count = 0
        imsize = 1

        try:
            if len(img.shape) == 3:
                # Ensure image is BGR before splitting
                if img.shape[2] == 4:  # RGBA
                    img = cv2_module.cvtColor(img, cv2_module.COLOR_RGBA2BGR)
                elif img.shape[2] == 1:  # Grayscale
                    img = cv2_module.cvtColor(img, cv2_module.COLOR_GRAY2BGR)
                # Now assume BGR
                b, g, r = cv2_module.split(img)
                wb, wg, wr = b == 255, g == 255, r == 255
                white_pixels = np_module.bitwise_and(wb, np_module.bitwise_and(wg, wr))
                white_count = np_module.sum(white_pixels)
                imsize = img.size / 3
            elif len(img.shape) == 2:  # Grayscale
                white_pixels = img == 255
                white_count = np_module.sum(white_pixels)
                imsize = img.size
            else:
                console.print(
                    f"Warning: Unexpected image shape {img.shape} in pct_white.",
                    style="warning",
                )
                return 1.0  # Treat unexpected shape as white

            return white_count / imsize if imsize > 0 else 0
        except Exception as e:
            console.print(f"Error in pct_white: {e}", style="danger")
            return 1.0  # Treat errors as white

    @staticmethod
    def simple_plot(img, plt_module, cv2_module, console):
        """Plot img using matplotlib.pyplot"""
        if plt_module is None:
            console.print("Error: Matplotlib not loaded.", style="danger")
            return
        if cv2_module is None:
            console.print("Error: OpenCV not loaded.", style="danger")
            return
        plt_module.imshow(
            cv2_module.cvtColor(img, cv2_module.COLOR_BGR2RGB)
            if len(img.shape) == 3
            else img
        )
        plt_module.show()

    @staticmethod
    def plot_all_boxes(img, boxes, cv2_module, np_module, console):
        """Plots all rectangles from boxes onto img."""
        if cv2_module is None or np_module is None:
            console.print("Error: OpenCV or NumPy not loaded.", style="danger")
            return img  # Return original image on error
        copy = img.copy()
        alpha = 0.4
        for box in boxes:
            x, y, w, h = box.x, box.y, box.width, box.height
            rand_color = list(np_module.random.random(size=3) * 256)
            cv2_module.rectangle(copy, (x, y), (x + w, y + h), rand_color, -1)
        return cv2_module.addWeighted(copy, alpha, img, 1 - alpha, 0)

    @staticmethod
    def remove_duplicate_bboxes(boxes):
        """Remove bounding boxes from a list that start at the same y-coord"""
        new = []
        seen_y = set()
        for box in boxes:
            if box.y not in seen_y:
                new.append(box)
                seen_y.add(box.y)
        return new

    @staticmethod
    def merge_bboxes(lst):
        """Merge overlapping bounding boxes based on vertical position."""
        if not lst:
            return []

        lst.sort(key=lambda box: box.y)

        merged = [lst[0]]
        for curr_box in lst[1:]:
            last_box = merged[-1]
            if curr_box.y < last_box.y + last_box.height:
                last_box.height = (
                    max(last_box.y_bottom, curr_box.y_bottom) - last_box.y
                )
                last_box.y_bottom = last_box.y + last_box.height
            else:
                merged.append(curr_box)

        return merged

    @staticmethod
    def expand_bbox(box, expand_factor, np_module, console, BBox_class):
        """Expand a bounding box by the given factor."""
        if np_module is None:
            console.print("Error: NumPy not loaded.", style="danger")
            return box  # Return original box or handle error appropriately
        x, y, w, h = box.x, box.y, box.width, box.height
        expansion = int(min(h, w) * expand_factor)
        x = max(0, x - expansion)
        y = max(0, y - expansion)
        h, w = h + (2 * expansion), w + (2 * expansion)
        return BBox_class(x, y, w, h)

    @staticmethod
    def get_file_name(path):
        """Extract filename without extension using regex."""
        filename = os.path.basename(path)
        filename = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)
        # Updated regex to handle trailing dots correctly
        match = re.match(r"^(.+?)(\.[^.]+)?$", filename.rstrip("."))
        if match:
            return match.group(1)
        return filename.rstrip(".")  # Fallback for names without extensions

    @staticmethod
    def escape_special_chars(s):
        """Return string s with LaTeX special characters escaped."""
        if not isinstance(s, str):
            s = str(s)

        # Process character by character instead of using string replace
        result = []
        for char in s:
            if char == "&":
                result.append(r"\&")
            elif char == "%":
                result.append(r"\%")
            elif char == "$":
                result.append(r"\$")
            elif char == "#":
                result.append(r"\#")
            elif char == "_":
                result.append(r"\_")
            elif char == "{":
                result.append(r"\{")
            elif char == "}":
                result.append(r"\}")
            elif char == "~":
                result.append(r"\textasciitilde{}")
            elif char == "^":
                result.append(r"\textasciicircum{}")
            elif char == "\\":
                result.append(r"\textbackslash{}")
            else:
                result.append(char)

        return "".join(result)

    @staticmethod
    def make_strlist(lst):
        """Make all the items of a lst a string"""
        return [str(i) for i in lst]

    @staticmethod
    async def async_write_all(filename, lst, console):
        """Asynchronously write all strings in lst to filename"""
        try:
            async with asyncio.Lock():
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w", encoding="utf-8") as f:
                    for s in lst:
                        f.write(s)
                        f.write("\n")
            if console:
                console.print(f"Wrote {len(lst)} strings to {filename}", style="info")
            return True
        except Exception as e:  # pylint: disable=broad-except
            if console:
                console.print(f"Error writing to {filename}: {e}", style="danger")
            return False

    @staticmethod
    def write_all(filename, content, console_instance=None):  # Accept console optionally
        """Write content to a file, handling potential errors."""
        _console = console_instance or console  # Use passed console or default
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            _console.print(
                f"Successfully wrote {len(content)} lines to {filename}", style="success"
            )
        except (IOError, OSError) as e:
            _console.print(f"Error writing file {filename}: {e}", style="danger")
        except Exception as e:  # pylint: disable=broad-except
            _console.print(f"Unexpected error writing file {filename}: {e}", style="danger")

    @staticmethod
    def create_latex_project_structure(base_path, pdf_name):
        """Create a proper LaTeX project directory structure."""
        # If base_path is absolute, use it directly; else join with cwd
        if os.path.isabs(base_path):
            safe_base = base_path
        else:
            safe_base = Utils.safe_join(os.getcwd(), base_path)
        safe_pdf_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", pdf_name)
        project_dir = Utils.safe_join(safe_base, safe_pdf_name)
        os.makedirs(project_dir, exist_ok=True)

        figures_dir = Utils.safe_join(project_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        build_dir = Utils.safe_join(project_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        return {
            "project_dir": project_dir,
            "figures_dir": figures_dir,
            "build_dir": build_dir,
        }

    @staticmethod
    async def extract_images_from_pdf(pdf_path, output_dir, executor, fitz_module, console):
        """Extract complete images from a PDF file asynchronously."""
        if fitz_module is None:
            if console:
                console.print("Error: PyMuPDF (fitz) not loaded.", style="danger")
            return []
        loop = asyncio.get_running_loop()
        # Ensure output_dir is absolute or relative to cwd before safe_join
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        safe_output_dir = output_dir
        os.makedirs(safe_output_dir, exist_ok=True)  # Ensure it exists

        def _extract_images():
            extracted_images = []
            try:
                doc = fitz_module.open(pdf_path)
                img_count = 0

                for page_num, page in enumerate(doc):
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        try:
                            base_image = doc.extract_image(xref)
                            if base_image and base_image["image"]:
                                image_ext = base_image["ext"]
                                image_bytes = base_image["image"]
                                img_filename = (
                                    f"embedded_image_p{page_num+1}_{img_index}.{image_ext}"
                                )
                                img_path = Utils.safe_join(safe_output_dir, img_filename)
                                with open(img_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                extracted_images.append(img_path)
                                img_count += 1
                        except Exception as img_extract_error:  # pylint: disable=broad-except
                            if console:
                                console.print(
                                    f"Error extracting image xref {xref} on page {page_num+1}: {img_extract_error}",
                                    style="warning",
                                )
                doc.close()
                return extracted_images
            except (FileNotFoundError, PermissionError) as file_error:
                if console:
                    console.print(
                        f"Error opening or accessing PDF {pdf_path}: {file_error}",
                        style="danger",
                    )
                return []
            except fitz_module.fitz.FitzError as fitz_error:
                if console:
                    console.print(
                        f"Fitz error processing PDF {pdf_path}: {fitz_error}",
                        style="danger",
                    )
                return []
            except Exception as e:  # pylint: disable=broad-except
                if console:
                    console.print(
                        f"Unexpected error extracting images from PDF {pdf_path}: {e}",
                        style="danger",
                    )
                return []

        return await loop.run_in_executor(executor, _extract_images)

    @staticmethod
    def segment(img, cv2_module, np_module, console, BBox_class=BBox):  # Add BBox_class default
        """Input: cv2 image of page. Output: BBox objects for content blocks."""
        if cv2_module is None:
            if console:
                console.print("Error: OpenCV not loaded.", style="danger")
            return []
        if np_module is None:
            if console:
                console.print("Error: NumPy not loaded.", style="danger")
            return []
        if img is None or img.size == 0:
            if console:
                console.print("Warning: Empty image passed to segment.", style="warning")
            return []

        img_height, img_width = img.shape[:2]

        try:
            gray = cv2_module.cvtColor(img, cv2_module.COLOR_BGR2GRAY)
            img_bw = cv2_module.adaptiveThreshold(
                gray,
                255,
                cv2_module.ADAPTIVE_THRESH_MEAN_C,
                cv2_module.THRESH_BINARY_INV,
                11,
                5,
            )

            k1 = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (3, 3))
            m1 = cv2_module.morphologyEx(img_bw, cv2_module.MORPH_GRADIENT, k1)

            k2 = cv2_module.getStructuringElement(
                cv2_module.MORPH_RECT, (HORIZONTAL_POOLING, 5)
            )
            m2 = cv2_module.morphologyEx(m1, cv2_module.MORPH_CLOSE, k2)

            k3 = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (5, 5))
            m3 = cv2_module.dilate(m2, k3, iterations=2)

            contours, _ = cv2_module.findContours(
                m3, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
            )

            bboxes = []
            for c in contours:
                bx, by, bw, bh = cv2_module.boundingRect(c)

                if bh < MIN_TEXT_SIZE or bw < MIN_TEXT_SIZE:  # Add width check
                    continue

                block_slice = img[by : by + bh, bx : bx + bw]
                if (
                    Utils.pct_white(block_slice, np_module, cv2_module, console) >= 0.99
                ):  # Allow tiny non-white specks
                    continue

                bboxes.append(BBox_class(0, by, img_width, bh))

            return sorted(bboxes, key=lambda x: x.y)
        except cv2_module.error as cv_error:
            if console:
                console.print(
                    f"OpenCV error during segmentation: {cv_error}", style="danger"
                )
            return []
        except Exception as e:
            if console:
                console.print(f"Unexpected error during segmentation: {e}", style="danger")
            return []

    @staticmethod
    def process_bboxes(bboxes):
        """Process list of BBox objects to remove redundancy."""
        if not bboxes:
            return []

        bboxes = Utils.remove_duplicate_bboxes(bboxes)
        bboxes = Utils.merge_bboxes(bboxes)

        for i, curr_box in enumerate(bboxes[:-1]):
            next_box = bboxes[i + 1]
            if curr_box.y_bottom > next_box.y:
                overlap = curr_box.y_bottom - next_box.y
                new_boundary = next_box.y + overlap / 2
                curr_box.height = max(1, int(new_boundary) - curr_box.y)
                curr_box.y_bottom = curr_box.y + curr_box.height

                next_box.height = max(1, next_box.y_bottom - int(new_boundary))
                next_box.y = int(new_boundary)

        return bboxes

    @staticmethod
    def find_content_blocks(img, cv2_module, np_module, console, BBox_class=BBox):  # Add BBox_class default
        """Find all content blocks in page."""
        segmented_bboxes = Utils.segment(img, cv2_module, np_module, console, BBox_class)
        return Utils.process_bboxes(segmented_bboxes)

    @staticmethod
    def extract_block_image(bbox, page_img, page_height, np_module, console):
        """Extract the block image from the page based on bbox."""
        if np_module is None:
            if console:
                console.print("Error: NumPy not loaded.", style="danger")
            return None  # Indicate error
        if page_img is None:
            if console:
                console.print(
                    "Error: page_img not provided to extract_block_image.", style="danger"
                )
            return None
        if page_height is None:
            if console:
                console.print(
                    "Warning: Height not provided to extract_block_image.", style="warning"
                )
            page_height = page_img.shape[0]

        if (
            bbox.y < 0
            or bbox.y_bottom > page_height
            or bbox.x < 0
            or bbox.x + bbox.width > page_img.shape[1]
        ):
            if console:
                console.print(
                    f"Warning: Invalid bbox coordinates for extraction - "
                    f"y:{bbox.y}, y_bottom:{bbox.y_bottom}, page_h:{page_height}, "
                    f"x:{bbox.x}, x_right:{bbox.x + bbox.width}, page_w:{page_img.shape[1]}",
                    style="warning",
                )
            y_start = max(0, bbox.y)
            y_end = min(page_height, bbox.y_bottom)
            x_start = max(0, bbox.x)
            x_end = min(page_img.shape[1], bbox.x + bbox.width)
            if y_start >= y_end or x_start >= x_end:
                return np_module.zeros((1, 1, 3), dtype=np_module.uint8)
            return page_img[y_start:y_end, x_start:x_end]

        y_start = max(0, bbox.y)
        y_end = min(page_height, bbox.y_bottom)
        x_start = max(0, bbox.x)
        x_end = min(page_img.shape[1], bbox.x + bbox.width)

        if y_start >= y_end or x_start >= x_end:
            if console:
                console.print(
                    f"Warning: Calculated empty slice in extract_block_image.",
                    style="warning",
                )
            return np_module.zeros((1, 1, 3), dtype=np_module.uint8)

        return page_img[y_start:y_end, x_start:x_end]
