"""Mostra a análise auxiliar sem confundir expressão com identidade ou certeza."""
import pandas as pd
import streamlit as st


def mostrar_expressao(resultado):
    if 'erro_expressao' in resultado:
        st.info(resultado['erro_expressao'])
        return
    expressao = resultado.get('expressao')
    if expressao is None:
        return
    with st.container(border=True):
        st.caption('EXPRESSÃO FACIAL · MODELO PRONTO FER+')
        st.subheader(expressao['nome'])
        st.write(f'Pontuação da categoria: **{expressao["pontuacao"]:.1%}**'.replace('.', ','))
        st.caption('Estimativa da expressão visível nesta foto. Não determina o que a pessoa está sentindo.')
    with st.expander('Como a expressão foi estimada'):
        st.image(expressao['imagem_cinza'], width=128, caption='Entrada do FER+: cinza, 64×64 pixels')
        st.write('O recorte alinhado passa por uma preparação separada: tons de cinza, '
                 'redimensionamento para 64×64 e valores de 0 a 255. O FER+ calcula oito saídas; '
                 'softmax produz as pontuações e selecionamos a maior.')
        st.dataframe(pd.DataFrame({
            'Expressão': expressao['nomes'], 'Pontuação (%)': expressao['pontuacoes'] * 100,
        }), hide_index=True, width='stretch',
            column_config={'Pontuação (%)': st.column_config.NumberColumn(format='%.2f')})
        st.caption('As pontuações comparam as oito categorias; não são certezas sobre sentimentos. '
                   'O modelo não foi treinado com o grupo e esta análise não interfere no nome previsto.')
