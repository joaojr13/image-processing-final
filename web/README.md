# Interface

`etapa_01_app.py` organiza as abas. `telas/` contém upload, imagens, resultados, textos, gráficos e estilo visual. Treinamento, inferência e leitura dos arquivos de dados ficam em `api`.

A interface usa `api.etapa_02_servicos` e `api.etapa_01_consultas` por chamadas Python locais. O cache e o estado da tela pertencem ao Streamlit e ficam aqui; as regras de processamento pertencem ao backend.

Execute na raiz do projeto, onde ficam `web`, `api` e `.streamlit`:

```sh
.venv/bin/python -m streamlit run web/etapa_01_app.py
```

O endereço permanece http://127.0.0.1:8505/. O arquivo `iniciar.command` na raiz executa o mesmo comando.

`web/requirements.txt` lista as dependências da interface. Como a aplicação utiliza o backend local, instale `requirements.txt` da raiz para executar o projeto completo.

O guia [ORDEM_EXECUCAO.md](../ORDEM_EXECUCAO.md) descreve a ordem de leitura das telas e a ligação com o backend.
