"""BBox class for representing bounding boxes in images"""


class BBox:
    """BBox object representing bounding rectangle (x, y, width, height)"""
    # pylint: disable=too-few-public-methods
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.y_bottom = self.y + self.height