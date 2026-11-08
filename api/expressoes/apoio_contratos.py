"""A expressão facial tem uma responsabilidade diferente da identidade."""
from abc import ABC, abstractmethod
from PIL import Image


class ErroAnaliseExpressao(RuntimeError):
    """Uma falha desta análise não deve apagar a identificação já calculada."""


class AnalisadorExpressao(ABC):
    @abstractmethod
    def analisar(self, rosto_alinhado: Image.Image) -> dict:
        """Recebe o recorte do rosto e estima uma categoria de expressão facial."""
        raise NotImplementedError
