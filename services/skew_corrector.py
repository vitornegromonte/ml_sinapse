import cv2
import numpy as np
from PIL import Image


class SkewCorrector:
    def __init__(self, min_angle: float = 0.5):
        self._min_angle = min_angle

    def correct(self, image: Image.Image) -> Image.Image:
        if image.width < 10 or image.height < 10:
            return image

        cv_img = np.array(image)
        if cv_img.size == 0:
            return image

        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                rect = cv2.minAreaRect(cnt)
                angle = rect[-1]
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))
        if abs(median_angle) < self._min_angle:
            return image

        h, w = cv_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            cv_img, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated)
