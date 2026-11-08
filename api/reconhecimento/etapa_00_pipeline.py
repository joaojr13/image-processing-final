"""PIPELINE COMPLETO: leia identificar() de cima para baixo para seguir a imagem.

Cada tratamento está nomeado e implementado em etapa_01_preparacao.py. As três superclasses
permitem distinguir localização, extração de características e identificação.
"""
import numpy as np
from api.expressoes.apoio_contratos import AnalisadorExpressao, ErroAnaliseExpressao
from .apoio_contratos import DetectorRosto, ExtratorCaracteristicas, ClassificadorIdentidade
from .etapa_01_preparacao import (
    carregar_imagem, corrigir_orientacao, converter_para_rgb,
    limitar_dimensoes_para_deteccao, alinhar_rosto_pelos_olhos,
    converter_para_tensor, redimensionar_rosto, normalizar_pixels,
)


def aplicar_softmax(pontuacoes_brutas):
    """Converte cinco saídas em pontuações que somam 1; não são certezas calibradas."""
    valores = np.asarray(pontuacoes_brutas, dtype=np.float32)
    exponenciais = np.exp(valores - valores.max(axis=1, keepdims=True))
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


def selecionar_integrante(pontuacoes, nomes):
    indice = int(pontuacoes[0].argmax())
    return nomes[indice]


class PipelineReconhecimento:
    def __init__(self, detector: DetectorRosto, extrator_congelado: ExtratorCaracteristicas,
                 classificador: ClassificadorIdentidade, nomes,
                 analisador_expressao: AnalisadorExpressao | None = None):
        self.detector = detector
        self.extrator_congelado = extrator_congelado
        self.classificador = classificador
        self.nomes = nomes
        self.analisador_expressao = analisador_expressao

    def identificar(self, conteudo_da_imagem):
        # 1. Ler e padronizar a foto recebida.
        imagem = carregar_imagem(conteudo_da_imagem)
        imagem_orientada = corrigir_orientacao(imagem)
        imagem_rgb = converter_para_rgb(imagem_orientada)
        imagem_preparada = limitar_dimensoes_para_deteccao(imagem_rgb)

        # 2. Localizar e alinhar um rosto. A MTCNN não identifica a pessoa.
        deteccao = self.detector.detectar(imagem_preparada)
        rosto_alinhado = alinhar_rosto_pelos_olhos(imagem_preparada, deteccao)

        # 3. Preparar os números que entram na rede externa.
        tensor_do_rosto = converter_para_tensor(rosto_alinhado)
        pixels_redimensionados = redimensionar_rosto(tensor_do_rosto)
        pixels_normalizados = normalizar_pixels(pixels_redimensionados)

        # 4. REDE PRONTA CONGELADA: pixels → 512 características.
        caracteristicas = self.extrator_congelado.extrair(pixels_normalizados)

        # 5. NOSSA REDE KERAS: 512 características → cinco valores brutos.
        pontuacoes_brutas = self.classificador.classificar(caracteristicas)
        pontuacoes = aplicar_softmax(pontuacoes_brutas)
        nome_previsto = selecionar_integrante(pontuacoes, self.nomes)

        # Os valores exibidos na interface vêm desta mesma execução.
        resultado = {
            'nome_previsto': nome_previsto,
            'imagem_preparada': imagem_preparada,
            'deteccao': deteccao,
            'rosto_alinhado': rosto_alinhado,
            'pixels_redimensionados': pixels_redimensionados[0].detach().numpy(),
            'caracteristicas': caracteristicas[0],
            'pontuacoes_brutas': pontuacoes_brutas[0],
            'pontuacoes': pontuacoes[0],
        }

        # 6. RAMO AUXILIAR: o mesmo recorte segue para uma rede pronta de expressão.
        # O nome previsto acima não depende desta análise nem é recalculado por ela.
        if self.analisador_expressao is not None:
            try:
                resultado['expressao'] = self.analisador_expressao.analisar(rosto_alinhado)
            except ErroAnaliseExpressao as erro:
                resultado['erro_expressao'] = str(erro)
        return resultado
