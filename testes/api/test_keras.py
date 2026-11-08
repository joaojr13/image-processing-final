"""Inferência, aprendizado e persistência dos pesos Keras."""
import numpy as np
import pytest
from api.apoio_configuracao_keras import keras
from api.reconhecimento.apoio_arquivos import PASTA_MODELOS
from api.reconhecimento.etapa_05_classificador_integrantes import ClassificadorIntegrantes
from api.treinamento.etapa_03_aprender import treinar_rede


def test_classificador_calcula_saidas_com_os_pesos_ativos():
    classificador = ClassificadorIntegrantes(PASTA_MODELOS / 'identificacao_pessoas.weights.h5')
    matriz, vieses = classificador.rede_identificacao_pessoas.get_layer('camada_de_saida').get_weights()
    entradas = np.random.default_rng(8).normal(0, 0.05, size=(7, 512)).astype(np.float32)
    # A Dense calcula a soma ponderada das características e acrescenta os vieses.
    esperado = entradas @ matriz + vieses
    np.testing.assert_allclose(classificador.classificar(entradas), esperado, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match='características'):
        classificador.classificar(np.zeros((1, 3, 96, 96)))


def test_keras_aprende_e_recarrega_os_pesos(tmp_path):
    rotulos = np.tile(np.arange(5), 4)
    caracteristicas = np.zeros((20, 512), dtype=np.float32)
    caracteristicas[np.arange(20), rotulos] = 1
    rede, curvas = treinar_rede(caracteristicas, rotulos, epocas=20)
    assert isinstance(rede, keras.Sequential)
    assert curvas[-1]['perda_treino'] < curvas[0]['perda_treino']
    caminho = tmp_path / 'treino.weights.h5'
    rede.save_weights(caminho)
    recarregada = ClassificadorIntegrantes(caminho)
    esperado = rede.predict(caracteristicas, verbose=0)
    np.testing.assert_allclose(recarregada.classificar(caracteristicas), esperado, atol=1e-6)
    np.testing.assert_array_equal(esperado.argmax(axis=1), rotulos)


def test_treino_recusa_classe_ausente():
    with pytest.raises(ValueError, match='cinco integrantes'):
        treinar_rede(np.zeros((4, 512)), np.arange(4), epocas=1)


def test_salvar_treinamento_registra_epocas_e_preserva_modelo_ativo(tmp_path, monkeypatch):
    import json
    from api.treinamento import etapa_04_salvar as gravacao
    from api.reconhecimento.apoio_arquivos import calcular_resumo_arquivo

    monkeypatch.setattr(gravacao, 'PASTA_MODELOS', tmp_path)
    ativo = tmp_path / 'identificacao_pessoas.weights.h5'
    ativo.write_bytes(b'modelo ativo preservado')
    rotulos = np.arange(5)
    caracteristicas = np.zeros((5, 512), dtype=np.float32)
    caracteristicas[np.arange(5), rotulos] = 1
    rede, curvas = treinar_rede(caracteristicas, rotulos, epocas=3, semente=7)
    registro_modelo = {'nomes': ['Diego', 'Fernanda', 'Guilherme', 'João', 'Samanta'],
                       'extrator': {'sha256': 'referencia_do_extrator'}}
    registros = [{'arquivo': f'dados/treino/exemplo_{indice}.png'} for indice in range(5)]
    caminho = gravacao.salvar_treinamento(rede, curvas, registro_modelo, registros, semente=7)
    registro = json.loads((caminho.parent / 'registro.json').read_text())
    assert registro['epocas'] == 3 and registro['semente'] == 7
    assert registro['imagens_treino'] == 5 and registro['curvas'] == curvas
    assert registro['pesos_sha256'] == calcular_resumo_arquivo(caminho)
    assert registro['arquivos_treino'] == [item['arquivo'] for item in registros]
    assert ativo.read_bytes() == b'modelo ativo preservado'
    recarregada = ClassificadorIntegrantes(caminho)
    np.testing.assert_allclose(recarregada.classificar(caracteristicas), rede.predict(caracteristicas, verbose=0), atol=1e-6)
