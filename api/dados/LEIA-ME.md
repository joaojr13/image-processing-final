# Dados

`treino/<integrante>/` contém 40 recortes RGB 96×96 alinhados, oito por pessoa. `teste/<integrante>/` contém cinco recortes, um por pessoa. `originais/` preserva as imagens antes do alinhamento. `manifesto.json` associa recorte, origem, pessoa, divisão e integridade.

O treino lê somente sua divisão; a avaliação percorre os originais de teste pelo pipeline completo. A aba Avaliação permite conferir os recortes. Fotos enviadas para identificação não são salvas nem acrescentadas à base.

Não distribua duplicatas ou quadros do mesmo vídeo entre treino e teste. A divisão original foi preservada e não garante sessões fotográficas independentes. Repetir avaliações não aumenta a quantidade de fotos de teste.

Alterar imagens exige atualizar consistentemente manifesto, pesos e avaliação. Os dados são ignorados pelo Git.
