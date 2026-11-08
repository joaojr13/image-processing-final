"""Verifica as três abas, mensagens e limpeza de resultados de outra imagem."""
from streamlit.testing.v1 import AppTest


def test_interface_sem_experimentos_e_com_codigo_real():
    pagina=AppTest.from_file('web/etapa_01_app.py',default_timeout=60).run()
    assert not pagina.exception,pagina.exception
    assert [aba.label for aba in pagina.tabs]==['Identificar pessoa','Como funciona','Avaliação']
    assert pagina.button[0].label=='Identificar pessoa'
    assert pagina.button[0].disabled
    assert len(pagina.selectbox)==1 # Só a galeria da avaliação; nenhum seletor de modelos.
    assert any('class RedeIdentificacaoPessoas' in codigo.value for codigo in pagina.code)
    assert any('camadas.Dropout(rate=0.1' in codigo.value for codigo in pagina.code)
    assert any('def executar()' in codigo.value and 'salvar_treinamento' in codigo.value for codigo in pagina.code)
    assert any('rede.compile(' in codigo.value and 'rede.fit(' in codigo.value for codigo in pagina.code)
    assert any(m.label=='Acurácia' and m.value=='100%' for m in pagina.metric)
    pagina.session_state['resultado']={'nome_previsto':'resultado antigo'}
    pagina.session_state['foto_processada']=('foto_anterior','modelo_anterior')
    pagina.run()
    assert not pagina.exception,pagina.exception
    assert 'resultado' not in pagina.session_state
