# Ordem de leitura e execução

Os arquivos `etapa_XX_...py` organizam a leitura de cada pasta. O número não faz o Python executar um arquivo automaticamente: quem determina a execução são as chamadas das funções. `etapa_00_pipeline.py` mostra a sequência completa da identificação. Arquivos `apoio_...py` configuram, montam ou descrevem componentes; não representam etapas consecutivas da foto.

Os nomes começam com `etapa_` para permitir imports normais do Python. `__init__.py` mantém o nome exigido pela estrutura dos pacotes. Os testes mantêm o prefixo `test_` usado pelo pytest.

## Interface — web

A entrada é `web/etapa_01_app.py`. Ela configura a página e organiza as abas.

| Arquivo em web/telas | Responsabilidade |
|---|---|
| `etapa_01_estilo.py` | Aplica a aparência da interface |
| `etapa_02_reconhecer.py` | Recebe a foto, solicita a análise à API e apresenta o resultado |
| `etapa_03_expressoes.py` | Apresenta a expressão estimada e suas pontuações dentro da tela de identificação |
| `etapa_04_como_funciona.py` | Explica o processamento e mostra o código real |
| `etapa_05_avaliacao.py` | Apresenta métricas, matriz e galeria |

As três últimas telas não são fases obrigatórias de processamento da imagem. A numeração acompanha a organização da interface. Os componentes das abas são construídos pelo Streamlit; a análise da foto só é solicitada ao clicar no botão de identificação.

## Entrada do backend — api

1. `etapa_01_consultas.py` fornece configurações e nomes; também atende às consultas de métricas, galeria e código.
2. `etapa_02_servicos.py` prepara a prévia e fornece o analisador de fotos à interface.

`apoio_configuracao_keras.py` seleciona o motor do Keras antes de importá-lo. A ordem dos imports garante essa configuração, independentemente da posição do arquivo na listagem.

## Reconhecimento — api/reconhecimento

Comece por **`etapa_00_pipeline.py`**, no método `PipelineReconhecimento.identificar`.

| Arquivo | Papel no percurso da imagem |
|---|---|
| `etapa_01_preparacao.py` | Leitura, orientação, RGB, limite de tamanho, alinhamento, tensor e normalização |
| `etapa_02_detector_mtcnn.py` | Localiza o rosto e os olhos |
| `etapa_03_extrator_facenet.py` | Transforma o rosto preparado em 512 características, com a rede congelada |
| `etapa_04_rede_identificacao_pessoas.py` | Define nossa arquitetura Keras: Dropout e Dense |
| `etapa_05_classificador_integrantes.py` | Carrega nossos pesos e executa a rede nas 512 características |

A preparação é reutilizada: primeiro orienta e padroniza a foto; depois da detecção, alinha o rosto e prepara os pixels. A arquitetura é construída durante a montagem e executada pelo classificador. Por isso, os arquivos não são executados integralmente uma única vez em ordem numérica.

O caminho real é:

```text
Foto → preparação inicial → MTCNN → alinhamento e normalização
     → FaceNet congelada → nossa rede Keras → softmax → nome previsto
```

`apoio_montagem.py` conecta os componentes e carrega as redes. `apoio_contratos.py` define as superclasses. `apoio_arquivos.py` centraliza caminhos e integridade dos pesos.

## Expressão — api/expressoes

O pipeline envia o recorte alinhado para uma análise auxiliar, separada da identificação:

1. `etapa_01_preparacao.py`: converte para cinza, redimensiona para 64×64 e prepara os valores numéricos; também contém o softmax das oito saídas.
2. `etapa_02_analisador_ferplus.py`: carrega FER+, chama a preparação, executa o modelo e seleciona a expressão estimada.

`apoio_contratos.py` define a interface do analisador e o erro tratado sem apagar a identificação.

## Treinamento — api/treinamento

`etapa_00_treinar.py` reúne o fluxo completo em quatro chamadas. O treinamento é executado por comando, não a cada foto enviada pelo site.

| Arquivo | O que faz |
|---|---|
| `etapa_00_treinar.py` | Coordena carregar, extrair, aprender e salvar |
| `etapa_01_dados.py` | Confere o manifesto e carrega somente a divisão solicitada |
| `etapa_02_extrair_caracteristicas.py` | Usa a FaceNet congelada e gera características e rótulos |
| `etapa_03_aprender.py` | Contém compile() e fit(): aqui nossa rede aprende |
| `etapa_04_salvar.py` | Grava pesos e registro em uma nova pasta |
| `etapa_05_avaliar.py` | Avalia fotos de teste em um comando separado, sem treinar |

`apoio_parametros.py` centraliza épocas, semente e parâmetros do otimizador. `apoio_validacao_treino.py` confere as entradas e calcula pesos por integrante. `apoio_metricas.py` calcula acurácia, precisão, recall, F1 e matriz a partir dos rótulos.

Na raiz do projeto:

```sh
.venv/bin/python -m streamlit run web/etapa_01_app.py
.venv/bin/python -m api.treinamento.etapa_00_treinar
.venv/bin/python -m api.treinamento.etapa_05_avaliar
```
