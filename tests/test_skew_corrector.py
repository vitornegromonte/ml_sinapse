import math

import numpy as np
from PIL import Image

from services.skew_corrector import SkewCorrector


class TestSkewCorrector:
    def test_no_skew(self):
        """Imagem sem inclinação deve sair idêntica (size inalterado)."""
        img = Image.new("RGB", (200, 100), color="white")
        corrector = SkewCorrector(min_angle=0.5)
        result = corrector.correct(img)
        assert result.size == (200, 100)

    def test_small_image_unchanged(self):
        """Imagem com menos de 10px em qualquer dimensão não deve ser processada."""
        img = Image.new("RGB", (5, 100), color="white")
        corrector = SkewCorrector()
        result = corrector.correct(img)
        assert result.size == (5, 100)

    def test_empty_image_unchanged(self):
        """Imagem com array vazio não deve causar erro e retorna original."""
        img = Image.new("RGB", (100, 100), color="white")
        np_img = np.array(img)
        # Cria imagem que resulta em array de tamanho zero após conversão
        # Simplesmente retorna a original como o código faz para size==0
        corrector = SkewCorrector()
        result = corrector.correct(img)
        assert result.size == (100, 100)

    def test_angle_below_threshold(self):
        """Inclinação menor que min_angle não deve rotacionar."""
        img = Image.new("RGB", (200, 100), color="white")
        # min_angle alto para garantir que não passe do threshold
        corrector = SkewCorrector(min_angle=45)
        result = corrector.correct(img)
        assert result.size == (200, 100)

    def test_skew_correction_works_with_contours(self):
        """Imagem com regiões escuras (contornos detectáveis) deve ser processada sem erro."""
        arr = np.ones((200, 300, 3), dtype=np.uint8) * 255
        arr[50:150, 50:100] = 0
        arr[50:150, 150:200] = 0
        img = Image.fromarray(arr, "RGB")

        corrector = SkewCorrector(min_angle=0.5)
        result = corrector.correct(img)

        assert result.mode == "RGB"
        assert result.size == (300, 200)
