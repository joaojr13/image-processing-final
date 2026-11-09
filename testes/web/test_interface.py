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


def test_entrada_encontra_api_sem_pythonpath_do_projeto(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    entrada = Path(__file__).resolve().parents[2] / 'web/etapa_01_app.py'
    codigo = (
        'from streamlit.testing.v1 import AppTest; '
        f'pagina = AppTest.from_file({str(entrada)!r}, default_timeout=60).run(); '
        'assert not pagina.exception, pagina.exception; '
        'assert len(pagina.tabs) == 3'
    )
    resultado = subprocess.run(
        [sys.executable, '-I', '-c', codigo], cwd=tmp_path,
        capture_output=True, text=True, timeout=90,
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr


def test_falha_ao_processar_foto_remove_resultado_anterior():
    pagina = AppTest.from_string('''
import streamlit as st
from unittest.mock import patch
from web.telas.etapa_02_reconhecer import processar_foto
st.session_state['resultado'] = {'nome_previsto': 'resultado anterior'}
with patch('web.telas.etapa_02_reconhecer.obter_pipeline', side_effect=ValueError('Nenhum rosto localizado')):
    processar_foto(b'foto', 'modelo', ('nova foto', 'modelo'))
''').run()
    assert not pagina.exception
    assert 'resultado' not in pagina.session_state
    assert pagina.error[0].value == 'Nenhum rosto localizado'
