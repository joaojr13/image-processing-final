"""REDE EXTERNA: arquitetura e pesos prontos, aprendidos na base VGGFace2.

A InceptionResnetV1 é implementada na biblioteca facenet-pytorch. Seu conjunto de
convoluções transforma pixels em 512 características. Não é a rede criada pelo grupo.
"""
import torch
from facenet_pytorch import InceptionResnetV1
from .apoio_contratos import ExtratorCaracteristicas


class ExtratorFaceNet(ExtratorCaracteristicas):
    def __init__(self, caminho_dos_pesos):
        rede_externa = InceptionResnetV1(pretrained=None, classify=False)
        pesos = torch.load(caminho_dos_pesos, map_location='cpu', weights_only=True)
        # A saída antiga da VGGFace2 não é usada; queremos somente as características.
        pesos = {nome: valor for nome, valor in pesos.items() if not nome.startswith('logits.')}
        rede_externa.load_state_dict(pesos, strict=True)
        self.rede_congelada = rede_externa
        self.rede_congelada.requires_grad_(False)  # Os pesos não são treináveis.
        self.rede_congelada.eval()  # Dropout e BatchNorm ficam em modo de avaliação.

    def extrair(self, pixels_normalizados):
        # Reforça o modo de avaliação antes de cada uso da rede externa.
        self.rede_congelada.eval()
        with torch.no_grad():  # Não constrói um caminho de aprendizado.
            caracteristicas = self.rede_congelada(pixels_normalizados)
        # Fronteira explícita: PyTorch externo → NumPy → nossa rede Keras.
        return caracteristicas.detach().cpu().numpy()  # [N,512], sem gradientes.
