"""BBox class for representing bounding boxes in images"""

class BBox:
    """BBox object representing bounding rectangle (x, y, width, height)"""
    def __init__(self, x, y, w, h):
        """Initializes the BBox object."""
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def __repr__(self):
        """String representation for debugging."""
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    # Optional: Add properties for convenience if needed (e.g., x2, y2)
    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h