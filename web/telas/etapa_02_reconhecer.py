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
    cor = '#B7791F' if resultado['nome_previsto'] == 'Desconhecido' else '#16736B'
    desenho.rectangle(resultado['deteccao'].caixa.tolist(), outline=cor, width=4)
    return imagem


def mostrar_resultado(resultado, nomes):
    with st.container(border=True):
        st.caption('IDENTIFICAÇÃO CONCLUÍDA')
        rosto, descricao = st.columns([1, 2])
        rosto.image(resultado['rosto_alinhado'], width=128)
        descricao.caption('Nome previsto')
        descricao.subheader(resultado['nome_previsto'])
        indice = int(resultado['pontuacoes'].argmax())
        descricao.write(f'Maior pontuação: **{resultado["pontuacoes"][indice]:.2%}**'.replace('.', ','))
        limiar = resultado['limiar_identificacao']
        if resultado['nome_previsto'] == 'Desconhecido':
            st.info(f'A maior pontuação não atingiu o mínimo de {limiar:.0%}.')
        st.caption(f'Abaixo de {limiar:.0%}: Desconhecido. A partir desse valor: nome mais pontuado. '
                   'O limiar não foi calibrado com pessoas externas; elas ainda podem receber pontuação alta.')
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


def receber_foto(identificador_modelo):
    """Recebe a imagem e limpa o resultado anterior quando a foto ou o modelo muda."""
    st.subheader('Escolha ou tire uma foto')
    st.write('Escolha uma foto de frente, bem iluminada e com apenas uma pessoa.')
    origem = st.segmented_control(
        'Origem da foto', ['Enviar foto', 'Usar webcam'],
        default='Enviar foto', key='origem_foto',
    ) or 'Enviar foto'
    if origem == 'Usar webcam':
        st.caption('Permita o acesso à câmera no navegador, tire a foto e selecione Identificar pessoa.')
        arquivo = st.camera_input('Tirar foto do rosto', key='foto_webcam')
    else:
        arquivo = st.file_uploader(
            'Selecionar foto do rosto', type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility='collapsed', key='foto_enviada',
        )
        st.caption('JPG, PNG ou WEBP · até 12 MB.')
    st.caption('A foto é usada somente nesta análise; não é salva no projeto nem adicionada ao treinamento.')
    conteudo = arquivo.getvalue() if arquivo is not None else None
    identificador_foto = hashlib.sha256(conteudo).hexdigest() if conteudo else None
    identificador_atual = (origem, identificador_foto, identificador_modelo)
    if st.session_state.get('foto_processada') != identificador_atual:
        st.session_state.pop('resultado', None)

    previa = None
    if conteudo:
        try:
            previa = preparar_previa_foto(conteudo)
        except ValueError as erro:
            st.error(str(erro))
    return conteudo, previa, identificador_atual


def processar_foto(conteudo, identificador_modelo, identificador_atual):
    """Executa a análise solicitada e guarda o resultado para as próximas atualizações da tela."""
    st.session_state.pop('resultado', None)
    try:
        with st.spinner('Identificando a pessoa e analisando a expressão…'):
            resultado = obter_pipeline(identificador_modelo).identificar(conteudo)
        st.session_state['resultado'] = resultado
        st.session_state['foto_processada'] = identificador_atual
    except (ValueError, OSError) as erro:
        st.error(str(erro))


def renderizar(registro, identificador_modelo):
    esquerda, direita = st.columns([1, 1.05], gap='large')
    with esquerda, st.container(border=True):
        conteudo, previa, identificador_atual = receber_foto(identificador_modelo)
        if st.button('Identificar pessoa', type='primary', disabled=previa is None, width='stretch'):
            processar_foto(conteudo, identificador_modelo, identificador_atual)

        resultado = st.session_state.get('resultado')
        if resultado:
            st.image(desenhar_deteccao(resultado), caption='Rosto localizado na foto', width='stretch')
        elif previa:
            st.image(previa, caption='Foto selecionada', width='stretch')

    with direita:
        if resultado:
            mostrar_resultado(resultado, registro['nomes'])
        else:
            with st.container(border=True):
                st.subheader('O resultado aparece aqui')
                st.write('Envie uma foto ou tire uma pela webcam e selecione **Identificar pessoa**. Você verá o rosto localizado, '
                         'o nome previsto e a expressão estimada.')
            st.caption('Integrantes cadastrados: ' + ' · '.join(registro['nomes']))
    st.caption('Quer entender o caminho da imagem? A aba “Como funciona” mostra cada etapa e separa as duas redes.')
