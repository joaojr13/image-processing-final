"""Monta as três implementações concretas; o percurso da foto está em etapa_00_pipeline.py."""
import torch
from .apoio_arquivos import PASTA_MODELOS, ler_registro_modelo, conferir_pesos
from .etapa_02_detector_mtcnn import DetectorMTCNN
from .etapa_03_extrator_facenet import ExtratorFaceNet
from .etapa_05_classificador_integrantes import ClassificadorIntegrantes
from .etapa_00_pipeline import PipelineReconhecimento


def montar_pipeline(incluir_expressao=False):
    torch.set_num_threads(4)
    registro = ler_registro_modelo()
    conferir_pesos(registro)
    detector = DetectorMTCNN()
    extrator = ExtratorFaceNet(PASTA_MODELOS / registro['extrator']['arquivo'])
    classificador = ClassificadorIntegrantes(PASTA_MODELOS / registro['identificacao']['arquivo'])
    analisador_expressao = None
    if incluir_expressao:
        from api.expressoes.etapa_02_analisador_ferplus import AnalisadorFERPlus
        analisador_expressao = AnalisadorFERPlus()
    return PipelineReconhecimento(
        detector, extrator, classificador, registro['nomes'], analisador_expressao
    )
