"""Entrada do site. As telas e as redes têm seus próprios módulos em português."""
from pathlib import Path
import sys

# Streamlit pode iniciar pela pasta /web. Incluímos a raiz para encontrar /api.
raiz_projeto = str(Path(__file__).resolve().parents[1])
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

import streamlit as st
from api.etapa_01_consultas import carregar_contexto_modelos
from web.telas.etapa_01_estilo import aplicar_estilo
from web.telas import (
    etapa_02_reconhecer as reconhecer,
    etapa_04_como_funciona as como_funciona,
    etapa_05_avaliacao as avaliacao,
)

st.set_page_config(page_title='Rostos · Identificação de integrantes', page_icon='◉', layout='wide')
aplicar_estilo()
st.markdown('<span class="selo">PROCESSAMENTO DE IMAGENS · AP02</span>', unsafe_allow_html=True)
st.title('Reconhecimento de rostos')
st.markdown('<p class="introducao">Uma foto, um rosto e um caminho claro até a identificação.</p>', unsafe_allow_html=True)

try:
    registro, identificador_modelo = carregar_contexto_modelos()
except (OSError, ValueError):
    st.error('Os arquivos do modelo não estão disponíveis. Consulte a preparação do projeto no README.')
    st.stop()

aba_identificacao, aba_funcionamento, aba_avaliacao = st.tabs(['Identificar pessoa', 'Como funciona', 'Avaliação'])
with aba_identificacao:
    reconhecer.renderizar(registro, identificador_modelo)
with aba_funcionamento:
    como_funciona.renderizar(registro)
with aba_avaliacao:
    avaliacao.renderizar(registro)
