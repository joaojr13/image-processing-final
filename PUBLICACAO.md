# Publicação no Streamlit Community Cloud

## Configuração

No Streamlit Community Cloud, selecione **Create app** e informe:

| Campo | Valor |
|---|---|
| Repository | `joaojr13/image-processing-final` |
| Branch | `main` |
| Main file path | `web/etapa_01_app.py` |
| Advanced settings → Python version | `3.11` |

Não é necessário preencher Secrets nesta versão. O repositório público inclui as fotos da base e os arquivos do classificador, conforme autorização para publicação. Quem acessa o repositório pode obter esses arquivos.

## O que acontece ao iniciar

1. O Streamlit instala `requirements.txt` da raiz. Esse arquivo inclui as listas do backend e da interface.
2. No Linux, PyTorch e torchvision usam os pacotes CPU oficiais, sem dependências CUDA.
3. A página abre com a configuração, as explicações e a avaliação incluídas no Git.
4. Na primeira identificação, o programa baixa FaceNet e FER+ caso ainda não existam no servidor. Cada arquivo é conferido por SHA-256 antes de ser usado.
5. Os modelos são reutilizados nas próximas identificações. Um reinício que apague os arquivos locais pode exigir novo download.

O treinamento não é executado ao abrir o site. A arquitetura e os pesos de identificação são os mesmos usados localmente. As fotos enviadas pelo usuário ficam somente na memória da sessão.

## Arquivos publicados

- `api/dados/`: manifesto, 40 recortes de treino, cinco de teste e imagens originais, para galeria e reprodução da avaliação.
- `api/modelos/identificacao_pessoas.weights.h5` e `registro.json`: classificador do grupo e sua procedência.
- `api/modelos/expressoes/registro.json`: fonte e integridade do FER+.
- `api/relatorios/avaliacao.json`: resultados do modelo ativo.

Pesos externos, treinos adicionais, caches, segredos e ambiente virtual ficam fora do Git. A configuração compartilhada não fixa endereço nem porta; `iniciar.command` faz isso apenas para execução local.

## Validação

A rotina `.github/workflows/verificar.yml` instala o projeto em Ubuntu com Python 3.11, verifica as dependências e executa os testes da rede e da interface. Ela começa sem os modelos externos e exercita seu carregamento automático.

Antes de apresentar, confirme que a rotina passou no GitHub, abra o aplicativo publicado e teste uma foto. O tamanho de 140 MB dos pesos em disco não corresponde ao consumo de memória das redes em execução. A estabilidade no limite de recursos do Community Cloud precisa ser confirmada no próprio serviço.

Se um download falhar, a identificação informa o erro. Se somente FER+ falhar, o nome previsto continua disponível e a expressão aparece como indisponível. Se o serviço encerrar o processo por memória, os registros do deploy ajudam a distinguir isso de erros do código.

Referências: [publicar](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) e [dependências](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
