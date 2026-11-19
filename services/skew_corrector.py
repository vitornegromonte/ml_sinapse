"""Straighten a tilted scan before OCR."""
import cv2
import numpy as np
from PIL import Image

# Smaller images carry no readable text and make the contour steps unreliable.
_MIN_SIDE_PX = 10

# Joins the letters of a line into one blob (wide and short), so each blob
# is a text line and its tilt is the tilt of the page.
_LINE_KERNEL_SIZE = (30, 5)

# Ignore blobs smaller than this (in square pixels): specks, not text lines.
_MIN_BLOB_AREA_PX2 = 100


class SkewCorrector:
    """Rotate an image so its text lines are horizontal.

    The tilt is the median angle of the text-line blobs. A page that is already
    straight, or has no text lines, is returned unchanged.

    Args:
        min_angle: tilts below this many degrees are ignored.
    """

    def __init__(self, min_angle: float = 0.5):
        self._min_angle = min_angle

    def correct(self, image: Image.Image) -> Image.Image:
        """Return the straightened image. The input must be RGB.

        Gray, palette and 1-bit images make OpenCV fail, so the caller converts
        them first (see OcrWorker._load_image).
        """
        if image.width < _MIN_SIDE_PX or image.height < _MIN_SIDE_PX:
            return image

        pixels = np.array(image)
        if pixels.size == 0:
            return image

        angle = self._median_text_angle(pixels)
        if angle is None or abs(angle) < self._min_angle:
            return image

        return Image.fromarray(self._rotate(pixels, angle))

    #  private

    @staticmethod
    def _median_text_angle(pixels: np.ndarray) -> float | None:
        """Median tilt in degrees of the text-line blobs, or None if there are none."""
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        # Otsu picks the threshold itself. Inverted, text becomes white on black.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, _LINE_KERNEL_SIZE)
        blobs = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(blobs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for contour in contours:
            if cv2.contourArea(contour) <= _MIN_BLOB_AREA_PX2:
                continue
            angle = cv2.minAreaRect(contour)[-1]
            # minAreaRect reports angles by its own convention. Fold them into
            # (-45, 45], the tilt away from horizontal.
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)

        return float(np.median(angles)) if angles else None

    @staticmethod
    def _rotate(pixels: np.ndarray, angle: float) -> np.ndarray:
        """Rotate around the center. Edges repeat the border color, not black."""
        height, width = pixels.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(
            pixels, matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
