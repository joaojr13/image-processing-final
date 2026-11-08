"""REDE EXTERNA: arquitetura e pesos prontos, aprendidos na base VGGFace2.

A InceptionResnetV1 é implementada na biblioteca facenet-pytorch. Seu conjunto de
convoluções transforma pixels em 512 características. Não é a rede criada pelo grupo.
"""
import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
from .apoio_contratos import ExtratorCaracteristicas


class RedeExternaCongelada(nn.Module):
    """Protege tanto os pesos quanto as estatísticas de BatchNorm da rede pronta."""
    def __init__(self, rede_externa):
        super().__init__()
        self.rede_externa = rede_externa
        self.rede_externa.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Nome/argumento exigidos pelo PyTorch. A rede externa sempre fica em avaliação.
        return super().train(False)

    def forward(self, pixels_normalizados):
        with torch.no_grad():
            return self.rede_externa(pixels_normalizados)


class ExtratorFaceNet(ExtratorCaracteristicas):
    def __init__(self, caminho_dos_pesos):
        rede_externa = InceptionResnetV1(pretrained=None, classify=False)
        pesos = torch.load(caminho_dos_pesos, map_location='cpu', weights_only=True)
        # A saída antiga da VGGFace2 não é usada; queremos somente as características.
        pesos = {nome: valor for nome, valor in pesos.items() if not nome.startswith('logits.')}
        rede_externa.load_state_dict(pesos, strict=True)
        self.rede_congelada = RedeExternaCongelada(rede_externa)

    def extrair(self, pixels_normalizados):
        caracteristicas = self.rede_congelada(pixels_normalizados)
        # Fronteira explícita: PyTorch externo → NumPy → nossa rede Keras.
        return caracteristicas.detach().cpu().numpy()  # [N,512], sem gradientes.
