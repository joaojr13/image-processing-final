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
    # O AppTest 1.50 exige uma lista para serializar o controle segmentado.
    pagina.get('button_group')[0].set_value(['Enviar foto']).run()
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


def test_webcam_so_aparece_quando_escolhida_e_limpa_resultado():
    pagina = AppTest.from_string('''
import streamlit as st
from web.telas.etapa_02_reconhecer import receber_foto
conteudo, previa, identificador = receber_foto('modelo')
st.button('Identificar pessoa', disabled=previa is None)
''').run()
    assert not pagina.exception
    assert len(pagina.get('file_uploader')) == 1
    assert not pagina.get('camera_input')
    pagina.session_state['resultado'] = {'nome_previsto': 'resultado anterior'}
    pagina.get('button_group')[0].set_value(['Usar webcam']).run()
    assert not pagina.exception
    assert len(pagina.get('camera_input')) == 1
    assert not pagina.get('file_uploader')
    assert pagina.button[0].disabled
    assert 'resultado' not in pagina.session_state


def test_foto_da_webcam_segue_para_o_mesmo_processamento():
    pagina = AppTest.from_string('''
from io import BytesIO
from unittest.mock import patch, Mock
from PIL import Image
import streamlit as st
from web.telas.etapa_02_reconhecer import receber_foto, processar_foto
foto = BytesIO()
Image.new('RGB', (200, 200), 'white').save(foto, format='JPEG')
with patch('web.telas.etapa_02_reconhecer.st.camera_input', return_value=foto):
    conteudo, previa, identificador = receber_foto('modelo')
if previa is not None:
    assert previa.size == (200, 200)
    assert conteudo == foto.getvalue()
    assert identificador[0] == 'Usar webcam'
    analisador = Mock()
    analisador.identificar.return_value = {'nome_previsto': 'resultado de teste'}
    with patch('web.telas.etapa_02_reconhecer.obter_pipeline', return_value=analisador):
        processar_foto(conteudo, 'modelo', identificador)
    analisador.identificar.assert_called_once_with(foto.getvalue())
''').run()
    pagina.get('button_group')[0].set_value(['Usar webcam']).run()
    assert not pagina.exception
    assert pagina.session_state['resultado']['nome_previsto'] == 'resultado de teste'
    # Trocar a origem invalida a análise, mesmo antes de enviar outra imagem.
    pagina.get('button_group')[0].set_value(['Enviar foto']).run()
    assert not pagina.exception
    assert 'resultado' not in pagina.session_state
