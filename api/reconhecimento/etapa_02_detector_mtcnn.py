"""Detector pronto: encontra rosto e olhos; não decide o nome do integrante."""
import numpy as np
from facenet_pytorch import MTCNN
from .apoio_contratos import DetectorRosto, DeteccaoRosto


class DetectorMTCNN(DetectorRosto):
    def __init__(self):
        self.rede_detectora = MTCNN(
            keep_all=True, select_largest=False, min_face_size=20, device='cpu'
        )

    def detectar(self, imagem_rgb):
        caixas, pontuacoes, pontos = self.rede_detectora.detect(imagem_rgb, landmarks=True)
        indices_validos = []
        if caixas is not None:
            for indice, pontuacao in enumerate(pontuacoes):
                if pontuacao is not None and float(pontuacao) >= 0.95:
                    indices_validos.append(indice)
        if not indices_validos:
            raise ValueError('Nenhum rosto foi localizado com segurança. Use uma foto de frente e bem iluminada.')
        if len(indices_validos) > 1:
            raise ValueError('A foto contém mais de um rosto. Nesta versão, envie uma foto com apenas uma pessoa.')
        indice = indices_validos[0]
        return DeteccaoRosto(
            caixa=np.asarray(caixas[indice], dtype=float),
            pontos_faciais=np.asarray(pontos[indice], dtype=float),
            pontuacao=float(pontuacoes[indice]),
        )
