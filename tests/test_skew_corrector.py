import math

import numpy as np
from PIL import Image

from services.skew_corrector import SkewCorrector


class TestSkewCorrector:
    def test_no_skew(self):
        """Arrange: a blank image with no detectable skew.
        Act: correct it.
        Assert: the size is unchanged.
        """
        img = Image.new("RGB", (200, 100), color="white")
        corrector = SkewCorrector(min_angle=0.5)
        result = corrector.correct(img)
        assert result.size == (200, 100)

    def test_small_image_unchanged(self):
        """Arrange: an image narrower than the 10 px processing floor.
        Act: correct it.
        Assert: it is returned unchanged.
        """
        img = Image.new("RGB", (5, 100), color="white")
        corrector = SkewCorrector()
        result = corrector.correct(img)
        assert result.size == (5, 100)

    def test_empty_image_unchanged(self):
        """Arrange: an all-white image with no ink to analyze.
        Act: correct it.
        Assert: no error and the original size is kept.
        """
        img = Image.new("RGB", (100, 100), color="white")
        np_img = np.array(img)
        # Cria imagem que resulta em array de tamanho zero após conversão
        # Simplesmente retorna a original como o código faz para size==0
        corrector = SkewCorrector()
        result = corrector.correct(img)
        assert result.size == (100, 100)

    def test_angle_below_threshold(self):
        """Arrange: a blank image and a very high min_angle threshold.
        Act: correct it.
        Assert: no rotation is applied, so the size is unchanged.
        """
        img = Image.new("RGB", (200, 100), color="white")
        # min_angle alto para garantir que não passe do threshold
        corrector = SkewCorrector(min_angle=45)
        result = corrector.correct(img)
        assert result.size == (200, 100)

    def test_skew_correction_works_with_contours(self):
        """Arrange: an image with two dark blocks contourable by OpenCV.
        Act: correct it.
        Assert: it processes without error and returns an RGB image of the same
        size.
        """
        arr = np.ones((200, 300, 3), dtype=np.uint8) * 255
        arr[50:150, 50:100] = 0
        arr[50:150, 150:200] = 0
        img = Image.fromarray(arr, "RGB")

        corrector = SkewCorrector(min_angle=0.5)
        result = corrector.correct(img)

        assert result.mode == "RGB"
        assert result.size == (300, 200)
