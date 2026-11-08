# Avaliação

`avaliacao.json` contém previsões das cinco fotos originais de teste, matriz e métricas do modelo ativo. SHA-256 associa o resultado aos pesos e ao manifesto; a interface omite métricas divergentes.

Cinco acertos em cinco fotos descrevem somente esse conjunto pequeno, já consultado no desenvolvimento. Não são um teste novo e cego. Não há avaliação de rejeição de desconhecidos.

`migracao_keras.json` documenta a equivalência dos pesos ativos após a conversão para Keras. `validacao_treinamento_keras.json` registra a verificação do treinamento em Keras. São registros de procedência, sem código antigo e sem acrescentar amostras de teste. Novas avaliações ficam junto dos pesos em `api/modelos/treinamentos/`.
