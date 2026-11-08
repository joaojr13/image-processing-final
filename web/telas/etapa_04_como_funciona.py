"""Explicações do que o programa faz, sem instruções de fala ou apresentação."""
import pandas as pd
import streamlit as st
from api.etapa_01_consultas import ler_codigo_explicativo


def renderizar():
    st.subheader('Da imagem ao nome')
    st.write('O programa localiza o rosto, extrai suas características e associa esses números aos integrantes.')
    st.markdown('''<div class="etapas">
      <div class="etapa"><span class="numero">01 · LOCALIZAR</span><b>MTCNN</b><p>Uma rede pronta encontra o rosto e a posição dos olhos. Ela não escolhe o nome da pessoa.</p></div>
      <div class="etapa"><span class="numero">02 · EXTRAIR</span><b>FaceNet congelada</b><p>Depois do alinhamento e da normalização, outra rede pronta transforma o rosto em 512 características.</p></div>
      <div class="etapa"><span class="numero">03 · IDENTIFICAR</span><b>Nossa rede</b><p>Um classificador treinado com os integrantes recebe as 512 características e calcula uma saída para cada nome.</p></div>
    </div>''', unsafe_allow_html=True)
    externa, propria = st.columns(2)
    with externa:
        with st.container(border=True):
            st.markdown('**Rede pré-treinada · congelada**')
            st.write('InceptionResnetV1, aprendida na base VGGFace2 por terceiros. Entrega 512 números que representam padrões do rosto.')
            st.caption('23.482.624 parâmetros fixos · nenhum atualizado com o grupo')
    with propria:
        with st.container(border=True):
            st.markdown('**RedeIdentificacaoPessoas · Keras**')
            st.write('Arquitetura definida aqui: Dropout → Dense(5), entrada 512. Aprende a associar as características aos cinco integrantes.')
            st.caption('2.565 parâmetros treináveis · uma camada com pesos aprendidos')
    st.caption('Keras define e treina nossa rede. PyTorch é o motor de cálculo do Keras e também executa '
               'MTCNN e FaceNet. A passagem da FaceNet para nossa rede usa um vetor NumPy de 512 características.')
    st.info('Nossa rede é um classificador sem convoluções. A parte convolucional que extrai características '
            'vem pronta e fica congelada. Os dois conjuntos de pesos são armazenados separadamente.')
    with st.expander('Todos os tratamentos da imagem', expanded=False):
        st.dataframe(pd.DataFrame([
            ['1. Abrir e orientar', 'Aplica a orientação EXIF para fotos de celular.'],
            ['2. Padronizar cores', 'Converte para três canais: vermelho, verde e azul.'],
            ['3. Limitar a imagem', 'Reduz proporcionalmente o lado maior a até 1.200 pixels para a detecção.'],
            ['4. Detectar com MTCNN', 'Localiza rosto e cinco pontos faciais. Exige exatamente um rosto com pontuação de detecção ≥ 0,95.'],
            ['5. Alinhar pelos olhos', 'Aplica rotação, escala e translação. Produz RGB 96×96 com os olhos em posições fixas; fora da imagem, usa preto.'],
            ['6. Converter em tensor', 'Reorganiza os pixels como [1,3,96,96] e divide por 255. São um rosto, três canais e altura/largura.'],
            ['7. Redimensionar', 'Interpolação bilinear de 96×96 para 160×160. Não recupera detalhes ausentes.'],
            ['8. Normalizar', 'Para os valores já entre 0 e 1: (valor × 255 − 127,5) / 128. A entrada fica aproximadamente entre −1 e 1.'],
            ['9. Extrair características', 'A FaceNet congelada transforma os pixels em 512 números.'],
            ['10. Identificar', 'Nossa rede calcula cinco valores brutos. Softmax transforma-os em pontuações; a maior indica o nome previsto.'],
        ], columns=['Etapa', 'O que acontece']), hide_index=True, width='stretch')
        st.caption('Os tratamentos usados são os listados acima. Não aplicamos Fourier, suavização adicional, '
                   'equalização de histograma ou conversão para cinza no ramo de identificação.')
    with st.expander('Arquitetura completa da nossa rede', expanded=True):
        st.code('512 características → Dropout(0.1) → Dense(5) → 5 valores brutos', language='text')
        st.dataframe(pd.DataFrame([
            ['Dropout', '1', '0', 'Desativa 10% das características aleatoriamente, somente no treino.'],
            ['Dense de saída', '1', '2.565', 'Combina as características com 2.560 pesos e cinco vieses.'],
        ], columns=['Camada', 'Quantidade', 'Parâmetros aprendidos', 'Função']), hide_index=True, width='stretch')
        st.write('São **duas camadas declaradas, uma com parâmetros treináveis**. A entrada de 512 valores '
                 'não é contada como outra camada. O softmax é aplicado depois da rede, durante a identificação.')
        st.caption('As linhas abaixo são lidas do arquivo real etapa_04_rede_identificacao_pessoas.py:')
        st.code(ler_codigo_explicativo('arquitetura'), language='python')
    with st.expander('Pipeline no código e responsabilidades'):
        st.code(ler_codigo_explicativo('pipeline'), language='python')
        st.write('As superclasses definem três operações: detectar, extrair e classificar. '
                 'As classes DetectorMTCNN, ExtratorFaceNet e ClassificadorIntegrantes implementam essas operações.')
    with st.expander('Como a nossa rede aprende'):
        st.write('Usamos 40 imagens rotuladas: oito por integrante. São dez fotos e 30 quadros de vídeos. '
                 'A FaceNet produz características fixas em NumPy; só a nossa rede Keras é treinada por 200 épocas com AdamW. '
                 'A entropia cruzada mede o erro, e o otimizador ajusta os 2.565 parâmetros. '
                 'As cinco fotos de teste não entram nesse ajuste.')
        st.write('compile() configura o otimizador AdamW e a entropia cruzada. '
                 'fit() percorre as épocas, ativa Dropout e ajusta os pesos automaticamente. '
                 'predict() identifica sem aprender e com Dropout desativado.')
        st.caption('Fluxo completo — etapa_00_treinar.py:')
        st.code(ler_codigo_explicativo('treinamento'), language='python')
        st.caption('Onde os pesos são aprendidos — etapa_03_aprender.py:')
        st.code(ler_codigo_explicativo('aprendizado'), language='python')
        st.caption('No treino, Dropout está ativo. Na identificação e na avaliação, fica desativado. '
                   'Os quadros de um mesmo vídeo permanecem juntos na divisão de treino.')
    with st.expander('Análise separada da expressão facial'):
        st.code('Rosto alinhado → cinza → 64×64 → FER+ pronto → 8 saídas → softmax', language='text')
        st.write('O mesmo rosto localizado pela MTCNN segue também para o FER+, uma CNN pronta '
                 'treinada por terceiros para classificar expressões faciais. Ela não recebe o nome previsto '
                 'nem as 512 características da FaceNet: recebe seus próprios pixels preparados.')
        st.write('Neste ramo, convertemos o recorte para cinza, redimensionamos para 64×64 e usamos '
                 'float32 na escala 0 a 255, com formato [1,1,64,64]. Não aplicamos a normalização da FaceNet.')
        st.write('As oito categorias são Neutra, Alegria, Surpresa, Tristeza, Raiva, Nojo, Medo e Desprezo. '
                 'A maior pontuação indica a categoria estimada para a expressão visível, '
                 'sem determinar o sentimento real. Não forçamos toda foto a alegria, tristeza, raiva ou medo.')
        st.caption('O FER+ não é treinado neste projeto. Seus pesos são separados e uma falha nele '
                   'não impede a identificação. Os 2.565 parâmetros da nossa rede permanecem os mesmos.')
        st.code(ler_codigo_explicativo('preparacao_expressao'), language='python')
        st.code(ler_codigo_explicativo('modelo_expressao'), language='python')
        st.markdown('[Documentação do FER+](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md)')
    st.markdown('[Documentação da rede pré-treinada](https://github.com/timesler/facenet-pytorch)')
