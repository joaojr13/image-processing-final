"""Verifica preparo, ordem das classes e independência em relação à identidade."""
from io import BytesIO
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from api.expressoes.etapa_02_analisador_ferplus import AnalisadorFERPlus, NOMES_EXPRESSOES
from api.expressoes.apoio_contratos import ErroAnaliseExpressao
from api.expressoes.etapa_01_preparacao import preparar_rosto_para_expressao, converter_saidas_em_pontuacoes
from api.reconhecimento.apoio_arquivos import RAIZ_API, PASTA_DADOS, PASTA_RELATORIOS
from api.reconhecimento.apoio_contratos import DeteccaoRosto
from api.reconhecimento.apoio_montagem import montar_pipeline
from api.reconhecimento.etapa_00_pipeline import PipelineReconhecimento


def test_preparo_ferplus_preserva_original_e_escala_0_a_255():
    original = Image.new('RGB', (96, 96), 'white')
    imagem_cinza, entrada = preparar_rosto_para_expressao(original)
    assert original.mode == 'RGB' and original.size == (96, 96)
    assert imagem_cinza.mode == 'L' and imagem_cinza.size == (64, 64)
    assert entrada.shape == (1, 1, 64, 64)
    assert entrada.dtype == np.float32
    assert np.all(entrada == 255)  # Não pode receber a normalização da FaceNet.


@pytest.mark.parametrize('indice, nome', [(1, 'Alegria'), (3, 'Tristeza'), (4, 'Raiva'), (6, 'Medo')])
def test_ordem_das_quatro_categorias_solicitadas(indice, nome):
    saida = np.full((1, 8), -1000., dtype=np.float32)
    saida[0, indice] = 1000
    pontuacoes = converter_saidas_em_pontuacoes(saida)
    assert NOMES_EXPRESSOES[int(pontuacoes.argmax())] == nome
    assert np.isfinite(pontuacoes).all()
    np.testing.assert_allclose(pontuacoes.sum(), 1)


def test_pesos_ausentes_geram_erro_explicito(tmp_path):
    analisador = AnalisadorFERPlus(tmp_path)
    with pytest.raises(ErroAnaliseExpressao, match='indisponível'):
        analisador.analisar(Image.new('RGB', (96, 96)))


def test_falha_de_expressao_nao_apaga_identidade():
    deteccao = DeteccaoRosto(np.array([0, 0, 96, 96]),
                            np.array([[30.72, 36.48], [65.28, 36.48]]), 1.0)
    detector = SimpleNamespace(detectar=lambda imagem: deteccao)
    extrator = SimpleNamespace(extrair=lambda pixels: np.zeros((1, 512), dtype=np.float32))
    classificador = SimpleNamespace(classificar=lambda numeros: np.array([[0., 0., 0., 10., 0.]], dtype=np.float32))

    class AnalisadorIndisponivel:
        def analisar(self, rosto):
            raise ErroAnaliseExpressao('Expressão indisponível no teste.')

    pipeline = PipelineReconhecimento(detector, extrator, classificador,
                                       ['Diego', 'Fernanda', 'Guilherme', 'João', 'Samanta'],
                                       AnalisadorIndisponivel())
    arquivo = BytesIO()
    Image.new('RGB', (96, 96), 'white').save(arquivo, format='PNG')
    resultado = pipeline.identificar(arquivo.getvalue())
    assert resultado['nome_previsto'] == 'João'
    assert 'erro_expressao' in resultado and 'expressao' not in resultado


def test_pipeline_real_mantem_identidade_e_executa_ferplus():
    pipeline = montar_pipeline(incluir_expressao=True)
    manifesto = json.loads((PASTA_DADOS / 'manifesto.json').read_text())
    referencia = json.loads((PASTA_RELATORIOS / 'avaliacao.json').read_text())
    por_arquivo = {item['arquivo']: item for item in referencia['previsoes']}
    for registro in manifesto['registros']:
        if registro['divisao'] != 'teste':
            continue
        resultado = pipeline.identificar((RAIZ_API / registro['original']).read_bytes())
        anterior = por_arquivo[registro['arquivo']]
        assert resultado['nome_previsto'] == anterior['nome_previsto']
        np.testing.assert_allclose(resultado['pontuacoes'], anterior['pontuacoes'], rtol=1e-4, atol=1e-6)
        assert 'erro_expressao' not in resultado
        expressao = resultado['expressao']
        assert expressao['nome'] in NOMES_EXPRESSOES
        assert expressao['pontuacoes'].shape == (8,)
        np.testing.assert_allclose(expressao['pontuacoes'].sum(), 1, atol=1e-6)
        assert expressao['imagem_cinza'].size == (64, 64)
    # Isto é uma verificação funcional; não afirma que as expressões estão corretas.
