# Pesos separados

- `extrator_facenet.pt`: rede externa pré-treinada na VGGFace2. O carregador descarta a antiga saída de identidades da VGGFace2 e utiliza a representação de 512 números. Seus 23.482.624 parâmetros utilizados ficam congelados.
- `identificacao_pessoas.weights.h5`: nossa camada Dense em Keras, com 2.565 parâmetros. Dropout existe na arquitetura, mas não possui pesos.
- `registro.json`: ordem dos nomes, origem, arquitetura, treino, limitações e SHA-256.
- `treinamentos/`: novas execuções. Não substituem automaticamente o modelo ativo.

A ordem das saídas é Diego, Fernanda, Guilherme, João e Samanta. Mudar essa ordem sem atualizar pesos e registros torna os nomes incorretos.

Os pesos ativos foram convertidos para Keras: matriz da Linear transposta para Dense, com vieses preservados. O programa usa arquivos locais e não depende da instalação antiga. Os pesos são ignorados pelo Git; mantenha-os ao copiar o projeto.

## Expressão facial

`expressoes/ferplus.onnx` contém a CNN FER+ pronta. `expressoes/registro.json` registra origem, hash e oito classes. É independente dos dois pesos de identificação acima; não é treinada com o grupo. A análise roda localmente com ONNX Runtime. O modelo espera cinza 64×64 em float32, escala 0 a 255.
