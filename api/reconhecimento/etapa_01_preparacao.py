"""Todos os tratamentos da imagem, em funções explícitas e sem aprendizado."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.nn import functional as funcoes_neurais

# Mesmas posições usadas para preparar as imagens do modelo validado.
OLHOS_DESTINO = np.array([[30.72, 36.48], [65.28, 36.48]])


def carregar_imagem(conteudo):
    """Abre até 12 MB e 24 megapixels; nenhuma imagem enviada é salva em disco."""
    if not conteudo or len(conteudo) > 12 * 1024 * 1024:
        raise ValueError('Envie uma imagem com até 12 MB.')
    try:
        with Image.open(BytesIO(conteudo)) as imagem:
            if imagem.width * imagem.height > 24_000_000:
                raise ValueError('Use uma imagem com até 24 megapixels.')
            imagem.load()
            return imagem.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as erro:
        raise ValueError('Não foi possível abrir a imagem. Envie um arquivo JPG, PNG ou WEBP válido.') from erro


def corrigir_orientacao(imagem):
    """Aplica a orientação EXIF de fotos de celular, antes de procurar o rosto."""
    return ImageOps.exif_transpose(imagem)


def converter_para_rgb(imagem):
    """Padroniza os canais vermelho, verde e azul. Não converte para cinza."""
    return imagem.convert('RGB')


def limitar_dimensoes_para_deteccao(imagem_rgb):
    """Limita o lado maior a 1.200 pixels, preservando a proporção, para a MTCNN."""
    imagem_limitada = imagem_rgb.copy()
    imagem_limitada.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    return imagem_limitada


def calcular_matriz_alinhamento(olhos):
    """Calcula rotação, escala uniforme e translação para posicionar os dois olhos.

    A linha entre os olhos fica horizontal e a distância entre eles fica fixa.
    É uma transformação geométrica 2D: não corrige pose 3D nem reconstrói o rosto.
    """
    olhos = np.asarray(olhos, dtype=float)
    if olhos.shape != (2, 2) or not np.isfinite(olhos).all():
        raise ValueError('As posições dos olhos são inválidas.')
    olhos = olhos[np.argsort(olhos[:, 0])]
    diferenca_x, diferenca_y = olhos[1] - olhos[0]
    distancia_quadrada = diferenca_x ** 2 + diferenca_y ** 2
    if distancia_quadrada < 4:
        raise ValueError('O rosto está pequeno demais para alinhar. Use uma foto mais próxima.')
    distancia_destino = OLHOS_DESTINO[1, 0] - OLHOS_DESTINO[0, 0]
    coeficiente_a = distancia_destino * diferenca_x / distancia_quadrada
    coeficiente_b = distancia_destino * diferenca_y / distancia_quadrada
    rotacao_escala = np.array([[coeficiente_a, coeficiente_b], [-coeficiente_b, coeficiente_a]])
    translacao = OLHOS_DESTINO[0] - rotacao_escala @ olhos[0]
    matriz = np.eye(3)
    matriz[:2, :2] = rotacao_escala
    matriz[:2, 2] = translacao
    return matriz


def alinhar_rosto_pelos_olhos(imagem_rgb, deteccao):
    """Recorta/alinha em RGB 96×96; regiões fora da foto recebem pixels pretos."""
    matriz = calcular_matriz_alinhamento(deteccao.pontos_faciais[:2])
    # Pillow pede o caminho inverso: pixel da saída → posição na imagem original.
    matriz_inversa = np.linalg.inv(matriz)
    return imagem_rgb.transform(
        (96, 96), Image.Transform.AFFINE, matriz_inversa[:2].reshape(-1),
        resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0),
    )


def converter_para_tensor(rosto_alinhado):
    """RGB 96×96 → números float32 em [1,3,96,96], inicialmente entre 0 e 1.

    O primeiro 1 é a quantidade de rostos. O 3 corresponde aos canais RGB.
    A divisão por 255 prepara a interpolação numérica usada no treinamento.
    """
    if rosto_alinhado.size != (96, 96) or rosto_alinhado.mode != 'RGB':
        raise ValueError('O rosto precisa estar alinhado em RGB 96×96.')
    pixels = np.asarray(rosto_alinhado, dtype=np.float32) / 255
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()


def redimensionar_rosto(tensor_do_rosto):
    """[N,3,96,96] → [N,3,160,160], por interpolação bilinear; não cria detalhes."""
    return funcoes_neurais.interpolate(
        tensor_do_rosto, size=(160, 160), mode='bilinear', align_corners=False
    )


def normalizar_pixels(pixels_redimensionados):
    """Aplica a normalização esperada pela FaceNet pronta, sem aprender nada.

    Os valores recebidos estão entre 0 e 1. Retornamos à escala 0–255 para usar
    (pixel − 127,5) / 128. A saída fica aproximadamente entre −1 e +1.
    """
    if not torch.isfinite(pixels_redimensionados).all():
        raise ValueError('A imagem contém valores numéricos inválidos.')
    return (pixels_redimensionados * 255 - 127.5) / 128
