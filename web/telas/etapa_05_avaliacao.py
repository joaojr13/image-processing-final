"""Avaliação única do modelo em uso, sem comparações com experimentos anteriores."""
import altair as alt
import pandas as pd
import streamlit as st
from api.etapa_01_consultas import carregar_avaliacao, listar_imagens_da_base


def desenhar_matriz(matriz, nomes):
    linhas = [{'Pessoa real': nomes[linha], 'Nome previsto': nomes[coluna], 'Fotos': matriz[linha][coluna]}
              for linha in range(len(nomes)) for coluna in range(len(nomes))]
    dados = pd.DataFrame(linhas)
    base = alt.Chart(dados).encode(
        x=alt.X('Nome previsto:N', sort=nomes, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('Pessoa real:N', sort=nomes),
    )
    quadrados = base.mark_rect(stroke='white', strokeWidth=3).encode(
        color=alt.Color('Fotos:Q', scale=alt.Scale(domain=[0,max(1,max(max(linha) for linha in matriz))],
                                                 range=['#EEF3F0','#16736B']), legend=None),
        tooltip=['Pessoa real:N','Nome previsto:N','Fotos:Q'],
    )
    numeros = base.mark_text(fontSize=18, fontWeight=600).encode(
        text='Fotos:Q', color=alt.condition(alt.datum.Fotos > 0, alt.value('white'), alt.value('#728A80'))
    )
    st.altair_chart((quadrados + numeros).properties(height=300), use_container_width=True)


def renderizar(registro):
    st.subheader('Como avaliamos este modelo')
    st.write('As fotos de teste ficaram fora do treinamento da nossa rede. Aqui estão os resultados '
             'do modelo em uso, executando o caminho completo desde a foto original.')
    st.caption('Estas métricas avaliam somente os nomes. A expressão facial não foi avaliada: '
               'nossa base não possui rótulos de expressão conferidos.')
    st.caption('Treinamento: 40 imagens, oito por integrante · Teste: cinco fotos, uma por integrante')
    relatorio = carregar_avaliacao(registro)
    if relatorio is None:
        st.info('A avaliação deste modelo ainda não está disponível ou precisa ser atualizada.')
        return
    colunas = st.columns(4)
    colunas[0].metric('Acurácia', f'{relatorio["acuracia"]:.0%}')
    colunas[1].metric('Precisão média', f'{relatorio["precisao_macro"]:.0%}')
    colunas[2].metric('Recall médio', f'{relatorio["recall_macro"]:.0%}')
    colunas[3].metric('F1 médio', f'{relatorio["f1_macro"]:.2f}'.replace('.', ','))
    st.caption('“Média” usa a média simples das métricas dos cinco integrantes (macro).')
    matriz, detalhes = st.columns([1.2,1], gap='large')
    with matriz:
        with st.container(border=True):
            st.markdown('**Matriz de confusão**')
            desenhar_matriz(relatorio['matriz_confusao'], registro['nomes'])
            st.caption('Linhas: pessoa real. Colunas: nome previsto. A diagonal mostra os acertos. '
                       'Cada número é uma foto; o total da matriz é cinco.')
    with detalhes:
        with st.container(border=True):
            st.markdown('**Resultado de cada foto**')
            st.dataframe(pd.DataFrame([{'Pessoa real': item['nome_real'], 'Nome previsto': item['nome_previsto'],
                                       'Acertou': 'Sim' if item['acertou'] else 'Não'}
                                      for item in relatorio['previsoes']]), hide_index=True, width='stretch')
            st.write('**O que esse resultado permite dizer?**')
            st.write('O modelo acertou essas fotos reservadas. A base é pequena e as fotos já foram '
                     'usadas durante o desenvolvimento; o resultado não garante acerto em novas capturas.')
    with st.expander('Entender as métricas e os limites do teste'):
        st.markdown('''- **Acurácia:** proporção de fotos com o nome correto.
- **Precisão:** quando a rede escolheu um nome, quantas vezes estava certa.
- **Recall:** entre as fotos de uma pessoa, quantas receberam o nome correto.
- **F1:** combina precisão e recall; fica alto quando os dois são altos.''')
        st.dataframe(pd.DataFrame(relatorio['por_pessoa']).rename(columns={
            'nome':'Integrante','precisao':'Precisão','recall':'Recall','f1':'F1','quantidade':'Fotos de teste'}),
            hide_index=True, width='stretch')
        st.write('Há uma foto por integrante no teste. Uma única mudança de acerto altera bastante as métricas. '
                 'A divisão separa arquivos, mas não assegura ocasiões de captura independentes. '
                 'Não há avaliação de pessoas desconhecidas nesta versão.')
    with st.expander('Ver as imagens usadas no projeto'):
        st.caption('Estes são os recortes RGB 96×96 registrados antes do redimensionamento e da normalização.')
        divisao = st.radio('Divisão da base', ['Treino', 'Teste'], horizontal=True)
        nome = st.selectbox('Integrante', registro['nomes'])
        imagens = listar_imagens_da_base(divisao.lower(), nome)
        colunas = st.columns(4)
        for indice, item in enumerate(imagens):
            tipo = 'Quadro de vídeo' if item['tipo']=='quadro_video' else 'Foto'
            colunas[indice % 4].image(item['conteudo'], caption=f'{tipo} {indice+1}', width='stretch')
