# Treinamento em etapas

O arquivo `etapa_00_treinar.py` mostra o fluxo completo. Seu trabalho é conectar quatro etapas:

```text
Carregar imagens de treino
          ↓
Extrair 512 características por rosto com a FaceNet congelada
          ↓
Treinar nossa rede Keras com essas características
          ↓
Salvar os novos pesos e o histórico
```

| Arquivo | Responsabilidade |
|---|---|
| `etapa_00_treinar.py` | Organiza as chamadas acima e informa onde os pesos foram salvos |
| `etapa_01_dados.py` | Confere a separação treino/teste, a integridade e abre as imagens |
| `etapa_02_extrair_caracteristicas.py` | Redimensiona, normaliza, extrai características e transforma nomes em rótulos de 0 a 4 |
| `etapa_03_aprender.py` | Cria nossa rede, configura o aprendizado com compile() e treina com fit() |
| `etapa_04_salvar.py` | Cria uma nova pasta e grava pesos, perdas e informações do treino |
| `etapa_05_avaliar.py` | Avalia as fotos de teste separadamente, sem alterar os pesos |

Em `etapa_03_aprender.py`, `compile()` define como medir o erro e como ajustar os pesos. `fit()` realiza os ajustes durante as épocas. Somente os 2.565 parâmetros da nossa rede são treinados; a FaceNet produz características fixas antes disso.

`apoio_parametros.py` reúne os valores do treino: 200 épocas, semente 42, taxa de aprendizado 0,03, decaimento dos pesos 0,001 e epsilon 1e-8. A extração usa lotes de oito rostos; o aprendizado usa todos os exemplos de treino em um lote por época.

`apoio_validacao_treino.py` concentra as conferências das entradas. Também equilibra o peso de cada integrante se as quantidades de imagens forem diferentes. Na base atual, com oito imagens por pessoa, todos os pesos valem 1.

Na raiz do projeto:

```sh
.venv/bin/python -m api.treinamento.etapa_00_treinar
```

O comando salva um novo arquivo `.weights.h5` e informa o comando exato para avaliá-lo. O modelo ativo da interface é preservado. Para avaliar o modelo ativo:

```sh
.venv/bin/python -m api.treinamento.etapa_05_avaliar
```
