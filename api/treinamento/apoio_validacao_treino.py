"""Conferências dos dados e equilíbrio entre integrantes, antes do aprendizado."""
import numpy as np


def validar_entradas(caracteristicas, rotulos, epocas):
    """Exige 512 números por exemplo e um rótulo inteiro de 0 a 4 por imagem."""
    if caracteristicas.ndim != 2 or caracteristicas.shape[1] != 512:
        raise ValueError('O treino recebe [N,512] características.')
    if rotulos.shape != (len(caracteristicas),):
        raise ValueError('Forneça um rótulo por imagem.')
    if not np.issubdtype(rotulos.dtype, np.integer):
        raise ValueError('Os rótulos precisam ser números inteiros.')
    if not np.isfinite(caracteristicas).all():
        raise ValueError('As características contêm valores inválidos.')
    if np.any((rotulos < 0) | (rotulos > 4)):
        raise ValueError('Os rótulos devem estar entre 0 e 4.')
    if len(np.unique(rotulos)) != 5:
        raise ValueError('Todos os cinco integrantes precisam de imagens de treino.')
    if epocas < 1:
        raise ValueError('O treinamento precisa de pelo menos uma época.')


def calcular_pesos_por_pessoa(rotulos):
    """Dá mais peso a integrantes com menos imagens. Na base equilibrada, todos valem 1."""
    contagens = np.bincount(rotulos, minlength=5)
    pesos = {}
    for pessoa, quantidade in enumerate(contagens):
        pesos[pessoa] = len(rotulos) / (5 * int(quantidade))
    return pesos
