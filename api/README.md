# Backend

Aqui ficam o processamento das fotos, as redes neurais, os pesos, os dados e a avaliação. Este pacote não importa Streamlit nem a interface `web`.

- `etapa_01_consultas.py`: entrega configurações, métricas, imagens da galeria e código para as explicações.
- `etapa_02_servicos.py`: cria o analisador de fotos e prepara a prévia validada.
- `reconhecimento/etapa_00_pipeline.py`: ordem completa dos tratamentos e da identificação.
- `reconhecimento/etapa_04_rede_identificacao_pessoas.py`: arquitetura inteira da nossa rede.
- `expressoes/`: ramo auxiliar de expressão facial com FER+.
- `treinamento/`: treinamento e cálculo das métricas.

A interface chama essas operações diretamente em Python. Esta pasta não é um servidor HTTP.

A partir da raiz do projeto:

```sh
.venv/bin/python -m api.treinamento.etapa_00_treinar
.venv/bin/python -m api.treinamento.etapa_05_avaliar
```

Os caminhos registrados no manifesto são relativos a esta pasta `api`. Por exemplo, `dados/teste/joao/foto_01.png` corresponde a `api/dados/teste/joao/foto_01.png` a partir da raiz do projeto. Isso preserva o manifesto e os hashes das avaliações após a reorganização.

Para instalar apenas as dependências do backend, execute na raiz:

```sh
.venv/bin/python -m pip install -r api/requirements.txt
```

Nossa rede é definida e treinada em Keras 3 (`Dropout` e `Dense`). `apoio_configuracao_keras.py` seleciona PyTorch como motor numérico. MTCNN e FaceNet permanecem externas e congeladas; a passagem entre o extrator e o classificador usa NumPy sem gradientes.

O guia [ORDEM_EXECUCAO.md](../ORDEM_EXECUCAO.md) explica a numeração, os arquivos de apoio e as chamadas entre as etapas.

O treinamento foi dividido em etapas menores. `treinamento/etapa_00_treinar.py` mostra o fluxo completo; `treinamento/etapa_03_aprender.py` contém o aprendizado com compile() e fit(). Consulte [treinamento/README.md](treinamento/README.md).
