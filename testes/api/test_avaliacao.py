"""Confere métricas com erros conhecidos e a avaliação a partir dos originais."""
import json
import pytest
from api.treinamento.apoio_metricas import calcular_metricas
from api.treinamento import etapa_05_avaliar as avaliacao
from api.reconhecimento.apoio_arquivos import PASTA_RELATORIOS, localizar_imagem


def test_metricas_com_acertos_erros_e_classe_sem_amostras():
    resultado = calcular_metricas([0, 0, 1], [0, 1, 1], ['Pessoa A', 'Pessoa B', 'Pessoa C'])
    assert resultado['matriz_confusao'] == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert resultado['acuracia'] == pytest.approx(2 / 3)
    assert resultado['precisao_macro'] == pytest.approx(0.5)
    assert resultado['recall_macro'] == pytest.approx(0.5)
    assert resultado['f1_macro'] == pytest.approx(4 / 9)
    assert resultado['por_pessoa'][2] == {
        'nome': 'Pessoa C', 'precisao': 0.0, 'recall': 0.0, 'f1': 0.0, 'quantidade': 0,
    }


@pytest.mark.parametrize('arquivo', ['../fora.png', '/tmp/fora.png'])
def test_caminho_da_imagem_nao_pode_sair_da_base(arquivo):
    with pytest.raises(ValueError, match='fora da pasta de dados'):
        localizar_imagem(arquivo)


def test_avaliacao_completa_preserva_relatorio_ativo(tmp_path, monkeypatch):
    caminho_ativo = PASTA_RELATORIOS / 'avaliacao.json'
    antes = caminho_ativo.read_bytes()
    esperado = json.loads(antes)
    monkeypatch.setattr(avaliacao, 'PASTA_RELATORIOS', tmp_path)
    resultado = avaliacao.executar()
    assert resultado['matriz_confusao'] == esperado['matriz_confusao']
    assert resultado['acuracia'] == esperado['acuracia']
    assert len(resultado['previsoes']) == 5
    assert all(item['acertou'] for item in resultado['previsoes'])
    assert json.loads((tmp_path / 'avaliacao.json').read_text()) == resultado
    assert caminho_ativo.read_bytes() == antes
