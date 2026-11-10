"""Limiar de identificação, rejeições na avaliação e apresentação de Desconhecido."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from api.reconhecimento.etapa_00_pipeline import selecionar_integrante, LIMIAR_IDENTIFICACAO
from api.reconhecimento.apoio_arquivos import PASTA_RELATORIOS, ler_registro_modelo
from api.treinamento import etapa_05_avaliar as avaliacao
from api import etapa_01_consultas as consultas


@pytest.mark.parametrize('tipo', [np.float32, np.float64])
def test_limiar_aceita_93_por_cento_e_recusa_valor_imediatamente_abaixo(tipo):
    limite = tipo(LIMIAR_IDENTIFICACAO)
    for valor, esperado in [
        (np.nextafter(limite, tipo(0)), 'Desconhecido'),
        (limite, 'Pessoa A'),
        (np.nextafter(limite, tipo(1)), 'Pessoa A'),
    ]:
        pontuacoes = np.array([[valor, 1 - valor]], dtype=tipo)
        assert selecionar_integrante(pontuacoes, ['Pessoa A', 'Pessoa B']) == esperado


def test_pipeline_rejeita_pontuacoes_baixas_sem_adicionar_saida_a_rede(monkeypatch):
    from api.reconhecimento.apoio_montagem import montar_pipeline
    from api.reconhecimento.apoio_arquivos import localizar_imagem
    from api.treinamento.etapa_01_dados import ler_registros_divisao

    pipeline = montar_pipeline()
    monkeypatch.setattr(pipeline.classificador, 'classificar', lambda entrada: np.zeros((1, 5)))
    imagem = ler_registros_divisao('teste')[0]
    resultado = pipeline.identificar(localizar_imagem(imagem['original']).read_bytes())
    assert resultado['nome_previsto'] == 'Desconhecido'
    assert resultado['limiar_identificacao'] == 0.93
    assert resultado['pontuacoes_brutas'].shape == (5,)
    np.testing.assert_allclose(resultado['pontuacoes'], [0.2] * 5)


def test_avaliacao_conta_rejeicao_de_integrante_como_erro(tmp_path, monkeypatch):
    pipeline = SimpleNamespace(identificar=lambda foto: {
        'nome_previsto': 'Desconhecido', 'pontuacoes': np.array([0.2] * 5),
    })
    monkeypatch.setattr(avaliacao, 'montar_pipeline', lambda: pipeline)
    monkeypatch.setattr(avaliacao, 'PASTA_RELATORIOS', tmp_path)
    resultado = avaliacao.executar()
    assert resultado['acuracia'] == 0
    assert resultado['recall_macro'] == 0
    assert resultado['f1_macro'] == 0
    assert all(not previsao['acertou'] for previsao in resultado['previsoes'])
    assert resultado['nomes_matriz'][-1] == 'Desconhecido'
    assert [linha[-1] for linha in resultado['matriz_confusao']] == [1, 1, 1, 1, 1, 0]
    assert sum(map(sum, resultado['matriz_confusao'])) == 5


@pytest.mark.parametrize('limiar_anterior', [None, 0.85])
def test_interface_omite_avaliacao_sem_o_limiar_atual(tmp_path, monkeypatch, limiar_anterior):
    relatorio = json.loads((PASTA_RELATORIOS / 'avaliacao.json').read_text())
    relatorio['limiar_identificacao'] = limiar_anterior
    (tmp_path / 'avaliacao.json').write_text(json.dumps(relatorio))
    monkeypatch.setattr(consultas, 'PASTA_RELATORIOS', tmp_path)
    assert consultas.carregar_avaliacao(ler_registro_modelo()) is None


def test_interface_mostra_desconhecido_e_as_cinco_pontuacoes():
    pagina = AppTest.from_string('''
import numpy as np
from PIL import Image
from web.telas.etapa_02_reconhecer import mostrar_resultado
mostrar_resultado({
    'nome_previsto': 'Desconhecido',
    'limiar_identificacao': 0.93,
    'pontuacoes': np.array([0.89, 0.04, 0.03, 0.02, 0.02]),
    'pontuacoes_brutas': np.zeros(5),
    'caracteristicas': np.zeros(512),
    'rosto_alinhado': Image.new('RGB', (96, 96)),
    'pixels_redimensionados': np.zeros((3, 160, 160)),
}, ['Diego', 'Fernanda', 'Guilherme', 'João', 'Samanta'])
''').run()
    assert not pagina.exception
    assert pagina.subheader[0].value == 'Desconhecido'
    assert '93%' in pagina.info[0].value
    assert len(pagina.dataframe[0].value) == 5
