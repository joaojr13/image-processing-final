# Interface

`etapa_01_app.py` organiza as abas. `telas/` contém upload, captura pela webcam, imagens, resultados, textos, gráficos e estilo visual. Treinamento, inferência e leitura dos arquivos de dados ficam em `api`.

A interface usa `api.etapa_02_servicos` e `api.etapa_01_consultas` por chamadas Python locais. O cache e o estado da tela pertencem ao Streamlit e ficam aqui; as regras de processamento pertencem ao backend.

Execute na raiz do projeto, onde ficam `web`, `api` e `.streamlit`:

```sh
.venv/bin/python -m streamlit run web/etapa_01_app.py
```

O comando direto usa a porta padrão do Streamlit. O arquivo `iniciar.command` na raiz fixa http://127.0.0.1:8505/ para uso local.

`web/dependencias.txt` lista as dependências da interface. Como a aplicação utiliza o backend local, instale `requirements.txt` da raiz para executar o projeto completo.

O guia [ORDEM_EXECUCAO.md](../ORDEM_EXECUCAO.md) descreve a ordem de leitura das telas e a ligação com o backend.

No Streamlit Cloud, use `web/etapa_01_app.py` como entrada. Existe apenas um `requirements.txt` da aplicação na raiz, para instalar também o backend. Consulte [PUBLICACAO.md](../PUBLICACAO.md).

Na aba **Identificar pessoa**, escolha **Enviar foto** ou **Usar webcam**. Para usar a webcam, permita o acesso à câmera no navegador, tire uma foto e selecione **Identificar pessoa**. As duas opções usam o mesmo processamento e limiar de identificação. A captura é uma foto estática; ela não é salva no projeto nem adicionada ao treinamento.
