"""Preparo exclusivo do FER+. Não muda os pixels usados pela identificação."""
import numpy as np
from PIL import Image


def preparar_rosto_para_expressao(rosto_alinhado):
    """RGB → cinza → 64×64 → float32 [1,1,64,64], na escala de 0 a 255.

    FER+ espera essa escala; NÃO usamos a normalização da FaceNet neste ramo.
    O primeiro 1 representa um rosto e o segundo representa um canal de cinza.
    """
    rosto_cinza = rosto_alinhado.convert('L')
    rosto_redimensionado = rosto_cinza.resize((64, 64), Image.Resampling.LANCZOS)
    pixels = np.asarray(rosto_redimensionado, dtype=np.float32)
    entrada = pixels[np.newaxis, np.newaxis, :, :]
    return rosto_redimensionado, entrada


def converter_saidas_em_pontuacoes(saidas):
    """Softmax estável: oito valores brutos → oito pontuações que somam 1."""
    valores = np.asarray(saidas, dtype=np.float32)
    if valores.shape != (1, 8) or not np.isfinite(valores).all():
        raise ValueError('O modelo de expressão retornou valores inválidos.')
    valores = valores[0]
    exponenciais = np.exp(valores - valores.max())
    return exponenciais / exponenciais.sum()
