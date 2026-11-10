# AP02 · Reconhecimento de rostos

Projeto didático em Streamlit para identificar **Diego, Fernanda, Guilherme, João e Samanta**, com uma foto e um rosto por execução. A interface separa identificação, funcionamento e avaliação.

## Organização: /web e /api

```text
ap02-image-processing/
├── web/
│   ├── etapa_01_app.py              # Entrada da interface
│   ├── telas/                       # Telas numeradas de 01 a 05
│   └── dependencias.txt
├── api/
│   ├── etapa_01_consultas.py        # Configuração e consultas
│   ├── etapa_02_servicos.py         # Operações de análise
│   ├── apoio_configuracao_keras.py  # Motor do Keras
│   ├── reconhecimento/             # Pipeline 00 e etapas 01 a 05
│   ├── expressoes/                 # Preparação 01 e análise 02
│   ├── treinamento/                # Fluxo 00 e etapas de treino 01 a 05
│   ├── dados/
│   ├── modelos/
│   ├── relatorios/
│   └── requirements.txt
├── testes/                         # Testes da API e da interface
├── .streamlit/config.toml
├── ORDEM_EXECUCAO.md                # Guia do percurso entre os arquivos
├── requirements.txt
└── iniciar.command
```

`web` cuida da interação e apresentação: escolhe a foto, mostra os resultados e desenha os gráficos. `api` prepara os pixels, executa as redes, treina e avalia. A interface usa somente `api.etapa_02_servicos` e `api.etapa_01_consultas` para acessar o backend. Os arquivos e as regras de validação ficam no backend.

A comunicação é **local, por chamadas Python no mesmo processo**. A pasta `api` não expõe endpoints HTTP nem inicia outro servidor. Essa organização mantém o projeto simples e permite executar treinamento e avaliação sem abrir a interface. O lançador local usa a porta 8505; na nuvem, a plataforma define o endereço e a porta.

Execute os comandos abaixo **na raiz do projeto**, que contém as pastas `web` e `api`.

Os prefixos `etapa_XX_` orientam a leitura do fluxo; os arquivos `apoio_` reúnem configuração, contratos e montagem. As chamadas das funções determinam a execução. Consulte [ORDEM_EXECUCAO.md](ORDEM_EXECUCAO.md) para acompanhar o percurso da imagem e distinguir identificação, expressão e treinamento.

## Executar

Nesta instalação, abra `iniciar.command` ou execute, na pasta do projeto:

```sh
.venv/bin/python -m streamlit run web/etapa_01_app.py
```

O lançador `iniciar.command` abre **http://127.0.0.1:8505/**; o comando direto usa a porta padrão do Streamlit. O endereço é local e funciona enquanto o processo estiver aberto.

Para montar outro ambiente, use Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run web/etapa_01_app.py
```

O repositório inclui a base registrada, os pesos do classificador, os registros e a avaliação ativa. FaceNet e FER+ são baixados das fontes originais quando necessários e conferidos por SHA-256. Os modelos externos ocupam cerca de 140 MB em disco; a primeira identificação pode demorar enquanto são baixados. As fotos enviadas ao site não são salvas.

Para publicar, siga [PUBLICACAO.md](PUBLICACAO.md). `requirements.txt` instala interface e backend; `web/dependencias.txt` é uma lista auxiliar. O arquivo `requirements-lock-macos.txt` documenta o ambiente local anterior, não é usado pela nuvem.

## Responsabilidades separadas

| Componente | Responsabilidade | Aprende com o grupo? |
|---|---|---|
| `DetectorMTCNN` | Localiza o rosto e os pontos faciais | Não; usa uma rede pronta |
| `ExtratorFaceNet` | Transforma o rosto preparado em 512 características | Não; InceptionResnetV1 pré-treinada na VGGFace2, congelada |
| `RedeIdentificacaoPessoas` | Associa as características aos cinco integrantes | Sim; somente seus 2.565 parâmetros são treinados |

**Nossa rede é um classificador sem convoluções.** As convoluções pertencem à rede externa pré-treinada. Essa distinção aparece no código, na interface e em arquivos separados de pesos. Não treinamos a CNN extratora com as fotos do grupo.

As superclasses abstratas `DetectorRosto`, `ExtratorCaracteristicas` e `ClassificadorIdentidade` ficam em `api/reconhecimento/apoio_contratos.py`. Definem as operações `detectar`, `extrair` e `classificar`. O pipeline recebe implementações dessas operações e as chama em ordem.

## Arquitetura da nossa rede

O arquivo **`api/reconhecimento/etapa_04_rede_identificacao_pessoas.py`** contém toda a arquitetura:

```text
512 características → Dropout(p=0.1) → Dense(5) → cinco valores brutos
```

São **duas camadas declaradas, uma treinável**. Dropout desativa aleatoriamente 10% das entradas no treino, não possui pesos e fica desativado na avaliação e na identificação. A camada Dense possui `512 × 5 + 5 = 2.565` parâmetros: 2.560 pesos e cinco vieses. Não há camada oculta. A entrada não é contada como camada adicional.

O softmax fica fora dessa classe, em `api/reconhecimento/etapa_00_pipeline.py`. Ele transforma os cinco valores em pontuações que somam 100%. A maior determina o candidato. Se for menor que 95%, o resultado é “Desconhecido”. A pontuação não é uma probabilidade calibrada de a pessoa realmente ter aquela identidade.

O extrator possui **23.482.624 parâmetros congelados**. `api/reconhecimento/etapa_03_extrator_facenet.py` usa `requires_grad_(False)` para congelar os pesos, `eval()` para manter o modo de avaliação e `torch.no_grad()` para extrair características sem aprendizado. O otimizador recebe exclusivamente os parâmetros da nossa rede.

## Pipeline da imagem

A ordem está em **`PipelineReconhecimento.identificar`**, no arquivo `api/reconhecimento/etapa_00_pipeline.py`. Cada tratamento possui uma função em `api/reconhecimento/etapa_01_preparacao.py`.

1. **Abrir e orientar:** valida o arquivo, limita a 12 MB e 24 megapixels e aplica a orientação EXIF.
2. **Converter para RGB:** mantém três canais de cor.
3. **Limitar dimensões:** reduz proporcionalmente o lado maior a até 1.200 pixels para a detecção.
4. **Localizar:** MTCNN encontra caixas e pontos faciais. Exige exatamente um rosto com pontuação de detecção de pelo menos 0,95.
5. **Alinhar:** os dois olhos determinam rotação, escala uniforme e translação. O recorte sai em RGB 96×96; áreas externas recebem preto. Não corrige pose tridimensional.
6. **Converter para tensor:** reorganiza os pixels em `[1,3,96,96]`, com valores entre 0 e 1. As dimensões são quantidade de rostos, canais, altura e largura.
7. **Redimensionar:** interpolação bilinear gera `[1,3,160,160]`, mantendo o tratamento do treinamento. Não recupera detalhes ausentes.
8. **Normalizar:** aplica `(valor × 255 − 127,5) / 128`, obtendo valores aproximadamente entre −1 e +1.
9. **Extrair:** FaceNet congelada produz `[1,512]` características.
10. **Identificar:** nossa rede produz `[1,5]` valores. Softmax gera as pontuações. A maior deve atingir 95% para aceitar o nome; abaixo disso, retorna “Desconhecido”.

No ramo de identificação, não aplicamos Fourier, equalização de histograma, suavização adicional nem conversão para cinza. As imagens intermediárias vêm da execução real. Os trechos mostrados em “Como funciona” são lidos dos arquivos reais do projeto.

## Dados e avaliação

| Divisão | Por integrante | Total | Origem |
|---|---:|---:|---|
| Treino | 8 | 40 | Duas fotos e seis quadros de vídeo por pessoa |
| Teste | 1 | 5 | Uma foto por pessoa |

Os nomes das pastas fornecem os rótulos; o modelo recebe pixels e rótulos numéricos, não os nomes dos arquivos. Os quadros de um mesmo vídeo permanecem no treino. O manifesto registra origem, divisão e SHA-256; o carregador impede duplicatas exatas e vídeos compartilhados entre treino e teste.

Os caminhos do manifesto são relativos à pasta `api`, que contém dados e modelos. A divisão original não garante sessões fotográficas independentes. As cinco fotos já foram usadas para avaliar versões anteriores: são uma verificação de regressão, não um teste novo e cego. Modificar modelos repetidamente com base nesses resultados limita a conclusão sobre generalização.

O modelo ativo percorreu o pipeline completo a partir das cinco fotos originais: **5/5 acertos**. Acurácia, precisão macro, recall macro e F1 macro são 100% **nessas cinco fotos**. Cada integrante tem uma amostra no teste; isso não demonstra desempenho perfeito em novas condições.

- **Acurácia:** proporção de previsões corretas.
- **Precisão:** entre as previsões de um nome, quantas estavam corretas.
- **Recall:** entre as fotos reais de um integrante, quantas foram reconhecidas.
- **F1:** média harmônica entre precisão e recall.
- **Macro:** média simples das métricas dos cinco integrantes.
- **Matriz de confusão:** linhas são nomes reais; colunas, nomes previstos. A diagonal contém os acertos.

**O limiar de identificação é 95%.** Uma pontuação inferior gera “Desconhecido”; igual ou superior aceita o nome mais pontuado. A constante `LIMIAR_IDENTIFICACAO` fica em `api/reconhecimento/etapa_00_pipeline.py`. É uma regra após a rede, que continua com cinco saídas. O limite não foi calibrado com pessoas externas e elas ainda podem ser aceitas. A matriz inclui rejeições; a média macro continua sobre os cinco integrantes. A base de teste contém somente integrantes e não mede a rejeição de pessoas externas.

## Treinar e avaliar

O arquivo `api/treinamento/etapa_00_treinar.py` coordena quatro chamadas: carregar os dados, extrair características, aprender e salvar. Cada responsabilidade tem seu próprio arquivo. O mapa completo está em [api/treinamento/README.md](api/treinamento/README.md).

Execute na raiz do projeto:

```sh
.venv/bin/python -m api.treinamento.etapa_00_treinar
```

O treinamento lê somente os 40 recortes alinhados de treino, já preparados com MTCNN e a transformação geométrica descrita acima. Redimensionamento e normalização usam as mesmas funções da identificação. FaceNet gera características fixas em NumPy; nossa rede Keras aprende durante 200 épocas, com entropia cruzada, AdamW, taxa de aprendizado 0,03, regularização de pesos 0,001 e semente 42.

Pesos, curva de perda e registro são salvos em `api/modelos/treinamentos/<data>/`. O programa informa o comando exato para avaliar os novos pesos. Não sobrescreve o modelo ativo.

```sh
.venv/bin/python -m api.treinamento.etapa_05_avaliar
.venv/bin/python -m api.treinamento.etapa_05_avaliar --pesos api/modelos/treinamentos/<data>/identificacao_pessoas.weights.h5
```

O primeiro comando avalia o modelo ativo e atualiza `api/relatorios/avaliacao.json`. O segundo salva o relatório junto dos pesos indicados. O relatório vincula métricas, pesos e manifesto por SHA-256; a interface omite métricas de outra versão.

Os pesos ativos foram aprendidos anteriormente em PyTorch e convertidos para Keras, com equivalência verificada. A procedência está em `api/modelos/registro.json`. Novos treinamentos usam Keras e salvam seus próprios pesos, sem substituir o modelo ativo.

## Encontrar o código

- **Pipeline:** `api/reconhecimento/etapa_00_pipeline.py`.
- **Arquitetura e Dropout:** `api/reconhecimento/etapa_04_rede_identificacao_pessoas.py`.
- **Treinamento, compile() e fit():** `api/treinamento/etapa_03_aprender.py`.
- **Mapa de todos os arquivos:** [ORDEM_EXECUCAO.md](ORDEM_EXECUCAO.md).

Nomes próprios do projeto, comentários e explicações estão em português. APIs das bibliotecas mantêm os nomes originais.

## Verificar

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Os testes cobrem arquitetura, congelamento, normalização, alinhamento, separação dos dados, ordem do pipeline, vínculo entre métricas e pesos e interface.

Keras define e treina nossa rede, usando PyTorch como motor numérico; `facenet-pytorch` fornece MTCNN e InceptionResnetV1; Pillow e NumPy tratam imagens; scikit-learn calcula métricas; Streamlit e pandas organizam a interface. Não usamos KNN nem SVM para escolher nomes.

A implementação externa e os pesos são provenientes de [facenet-pytorch](https://github.com/timesler/facenet-pytorch). Os termos aplicáveis à biblioteca, pesos e bases de terceiros continuam separados dos arquivos de autoria do grupo.


## Expressão facial

O mesmo rosto alinhado também segue para o **FER+**, uma CNN pronta e independente da identificação:

```text
                               ┌→ RGB 160×160 → FaceNet → nossa rede → nome
MTCNN → rosto alinhado RGB 96×96
                               └→ cinza 64×64 → FER+ pronto → expressão estimada
```

`api/expressoes/etapa_01_preparacao.py` converte o recorte para cinza, redimensiona para 64×64 e organiza float32 `[1,1,64,64]` na escala de 0 a 255. Este ramo usa sua própria preparação, sem a normalização da FaceNet.

`etapa_02_analisador_ferplus.py` executa o arquivo local `api/modelos/expressoes/ferplus.onnx` com ONNX Runtime, após verificar o registro. Não envia fotos para serviços externos. O modelo não é treinado com o grupo.

As oito saídas correspondem a **Neutra, Alegria, Surpresa, Tristeza, Raiva, Nojo, Medo e Desprezo**. Softmax gera pontuações e a maior indica a expressão estimada. A aparência não determina o sentimento real; não temos rótulos conferidos para medir a acurácia de expressão. As métricas do projeto avaliam somente a identidade.

A interface habilita esse ramo com `montar_pipeline(incluir_expressao=True)`. Falhas no FER+ preservam a identificação e exibem uma mensagem de indisponibilidade. Treinamento e avaliação da identidade não executam o FER+.

Fonte: [FER+ no ONNX Model Zoo](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md).

## Bibliotecas e execução da rede

Nossa arquitetura usa `keras.Sequential`. O projeto configura **Keras 3 com motor PyTorch** em `api/apoio_configuracao_keras.py`. MTCNN e FaceNet usam `facenet-pytorch`; FER+ usa ONNX Runtime.

A FaceNet entrega características NumPy sem gradientes. `compile()` configura o otimizador e a perda; `fit()` ajusta somente nossa rede, com Dropout ativo. `predict()` identifica sem aprender e com Dropout desativado. Os pesos são salvos por `save_weights()` em `.weights.h5` e restaurados por `load_weights()`.

Referências: [Keras](https://keras.io/getting_started/), [Sequential](https://keras.io/guides/sequential_model/) e [pesos](https://keras.io/api/models/model_saving_apis/weights_saving_and_loading/).
