"""Contratos simples: cada componente tem uma única responsabilidade pública.

As superclasses dizem O QUE cada componente entrega. As implementações concretas
mostram COMO isso é feito. A preparação dos pixels está em etapa_01_preparacao.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch


@dataclass
class DeteccaoRosto:
    """Resultado da localização: retângulo, cinco pontos faciais e pontuação."""
    caixa: np.ndarray
    pontos_faciais: np.ndarray
    pontuacao: float


class DetectorRosto(ABC):
    @abstractmethod
    def detectar(self, imagem_rgb: Image.Image) -> DeteccaoRosto:
        """Localiza um rosto e seus pontos faciais, sem identificar a pessoa."""
        raise NotImplementedError


class ExtratorCaracteristicas(ABC):
    @abstractmethod
    def extrair(self, pixels_normalizados: torch.Tensor) -> np.ndarray:
        """Recebe [N,3,160,160] e entrega [N,512] características fixas."""
        raise NotImplementedError


class ClassificadorIdentidade(ABC):
    @abstractmethod
    def classificar(self, caracteristicas_do_rosto: np.ndarray) -> np.ndarray:
        """Recebe [N,512] e entrega [N,5] valores brutos, um por integrante."""
        raise NotImplementedError
