"""Transforma as imagens de treino em 512 características por rosto, sem aprender."""
import numpy as np
import torch
from api.reconhecimento.apoio_arquivos import PASTA_MODELOS
from api.reconhecimento.etapa_03_extrator_facenet import ExtratorFaceNet
from api.reconhecimento.etapa_01_preparacao import redimensionar_rosto, normalizar_pixels
from .apoio_parametros import LOTE_EXTRACAO


def extrair_caracteristicas(imagens, registros_treino, registro_modelo):
    """Retorna as características e os rótulos numéricos na mesma ordem das imagens."""
    torch.set_num_threads(4)
    caminho = PASTA_MODELOS / registro_modelo['extrator']['arquivo']
    extrator_congelado = ExtratorFaceNet(caminho)

    # Os recortes já estão alinhados em RGB 96×96. Falta preparar a entrada da FaceNet.
    partes = []
    for inicio in range(0, len(imagens), LOTE_EXTRACAO):
        imagens_do_lote = imagens[inicio:inicio + LOTE_EXTRACAO]
        rostos_redimensionados = redimensionar_rosto(imagens_do_lote)
        pixels_normalizados = normalizar_pixels(rostos_redimensionados)
        caracteristicas_do_lote = extrator_congelado.extrair(pixels_normalizados)
        partes.append(caracteristicas_do_lote)
    caracteristicas = np.concatenate(partes)

    # A posição do nome no registro define o rótulo: 0, 1, 2, 3 ou 4.
    rotulos = []
    for imagem in registros_treino:
        rotulo = registro_modelo['rotulos'].index(imagem['pessoa'])
        rotulos.append(rotulo)
    return caracteristicas, np.array(rotulos)
