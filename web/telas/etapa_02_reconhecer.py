"""Tela inicial: uma foto, uma ação principal e um resultado em destaque."""
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from api.etapa_02_servicos import criar_analisador_fotos, preparar_previa_foto
from web.telas.etapa_03_expressoes import mostrar_expressao


@st.cache_resource
def obter_pipeline(identificador_modelo):
    # A chave inclui o registro do modelo; mudar pesos exige um registro atualizado.
    return criar_analisador_fotos()


def desenhar_deteccao(resultado):
    imagem = resultado['imagem_preparada'].copy()
    desenho = ImageDraw.Draw(imagem)
    desenho.rectangle(resultado['deteccao'].caixa.tolist(), outline='#16736B', width=4)
    return imagem


def mostrar_resultado(resultado, nomes):
    with st.container(border=True):
        st.caption('IDENTIFICAÇÃO CONCLUÍDA')
        rosto, descricao = st.columns([1, 2])
        rosto.image(resultado['rosto_alinhado'], width=128)
        descricao.caption('Nome previsto')
        descricao.subheader(resultado['nome_previsto'])
        indice = nomes.index(resultado['nome_previsto'])
        descricao.write(f'Pontuação: **{resultado["pontuacoes"][indice]:.1%}**'.replace('.', ','))
        st.caption('A pontuação compara os cinco integrantes. Pessoas externas também receberão um nome; '
                   'esta versão não possui rejeição validada de desconhecidos.')
    mostrar_expressao(resultado)
    with st.expander('Ver pontuações e processamento desta foto'):
        st.dataframe(pd.DataFrame({
            'Integrante': nomes,
            'Pontuação (%)': resultado['pontuacoes'] * 100,
        }), hide_index=True, width='stretch',
        column_config={'Pontuação (%)': st.column_config.NumberColumn(format='%.2f')})
        alinhado, redimensionado = st.columns(2)
        alinhado.image(resultado['rosto_alinhado'], caption='Alinhamento: RGB 96×96', width=150)
        pixels = (resultado['pixels_redimensionados'].transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
        redimensionado.image(Image.fromarray(pixels), caption='Entrada antes de normalizar: RGB 160×160', width=150)
        st.write('A rede congelada produziu 512 características. Nossa rede recebeu esses números e calculou '
                 'cinco valores brutos, convertidos em pontuações pelo softmax.')
        st.caption('Valores produzidos pela rede congelada — entrada da nossa rede:')
        st.line_chart(resultado['caracteristicas'], height=160)
        st.dataframe(pd.DataFrame({'Integrante': nomes, 'Valor bruto (logit)': resultado['pontuacoes_brutas']}),
                     hide_index=True, width='stretch')


def renderizar(registro, identificador_modelo):
    esquerda, direita = st.columns([1, 1.05], gap='large')
    with esquerda:
        with st.container(border=True):
            st.subheader('Envie uma foto')
            st.write('Escolha uma foto de frente, bem iluminada e com apenas uma pessoa.')
            arquivo = st.file_uploader('Selecionar foto do rosto', type=['jpg', 'jpeg', 'png', 'webp'],
                                       label_visibility='collapsed', key='foto_enviada')
            st.caption('JPG, PNG ou WEBP · até 12 MB · a foto enviada não é salva no projeto.')
            conteudo = arquivo.getvalue() if arquivo is not None else None
            identificador_foto = hashlib.sha256(conteudo).hexdigest() if conteudo else None
            identificador_atual = (identificador_foto, identificador_modelo)
            if st.session_state.get('foto_processada') != identificador_atual:
                st.session_state.pop('resultado', None)
            previa = None
            if conteudo:
                try:
                    previa = preparar_previa_foto(conteudo)
                except ValueError as erro:
                    st.error(str(erro))
            if st.button('Identificar pessoa', type='primary', disabled=previa is None, width='stretch'):
                st.session_state.pop('resultado', None)
                try:
                    with st.spinner('Identificando a pessoa e analisando a expressão…'):
                        resultado = obter_pipeline(identificador_modelo).identificar(conteudo)
                    st.session_state['resultado'] = resultado
                    st.session_state['foto_processada'] = identificador_atual
                except (ValueError, OSError) as erro:
                    st.error(str(erro))
            resultado = st.session_state.get('resultado')
            if resultado:
                st.image(desenhar_deteccao(resultado), caption='Rosto localizado na foto', width='stretch')
            elif previa:
                st.image(previa, caption='Foto selecionada', width='stretch')
    with direita:
        resultado = st.session_state.get('resultado')
        if resultado:
            mostrar_resultado(resultado, registro['nomes'])
        else:
            with st.container(border=True):
                st.markdown('''<div class="painel-vazio">
                <svg width="84" height="84" viewBox="0 0 84 84" fill="none" aria-hidden="true">
                  <rect x="1" y="1" width="82" height="82" rx="24" fill="#E7F0EB"/>
                  <path d="M23 31V23H31M53 23H61V31M61 53V61H53M31 61H23V53" stroke="#16736B" stroke-width="2.5" stroke-linecap="round"/>
                  <circle cx="42" cy="36" r="9" stroke="#16736B" stroke-width="2"/>
                  <path d="M29 57C29 44 55 44 55 57" stroke="#16736B" stroke-width="2" stroke-linecap="round"/>
                </svg><h3>O resultado aparece aqui</h3>
                <p>Envie a foto e selecione <b>Identificar pessoa</b>.<br>Você verá o rosto localizado, o nome previsto e a expressão estimada.</p>
                </div>''', unsafe_allow_html=True)
            st.caption('Integrantes cadastrados: ' + ' · '.join(registro['nomes']))
    st.caption('Quer entender o caminho da imagem? A aba “Como funciona” mostra cada etapa e separa as duas redes.')
