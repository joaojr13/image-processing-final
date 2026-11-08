"""Usa nossa rede Keras; seus pesos ficam separados da FaceNet e do FER+."""
import numpy as np
from .apoio_contratos import ClassificadorIdentidade
from .etapa_04_rede_identificacao_pessoas import RedeIdentificacaoPessoas


class ClassificadorIntegrantes(ClassificadorIdentidade):
    def __init__(self, caminho_dos_pesos):
        self.rede_identificacao_pessoas = RedeIdentificacaoPessoas()
        self.rede_identificacao_pessoas.load_weights(caminho_dos_pesos)

    def classificar(self, caracteristicas_do_rosto):
        caracteristicas = np.asarray(caracteristicas_do_rosto, dtype=np.float32)
        if caracteristicas.ndim != 2 or caracteristicas.shape[1] != 512:
            raise ValueError('Nossa rede recebe [N,512] características, não pixels.')
        # predict executa em inferência: Dropout desativado e nenhum aprendizado.
        return self.rede_identificacao_pessoas.predict(caracteristicas, verbose=0)
