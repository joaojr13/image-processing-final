"""Avalia as cinco fotos originais de teste pelo pipeline completo, sem treinar.

Executar: python -m api.treinamento.etapa_05_avaliar
A avaliação registra hashes dos modelos e dados para evitar mostrar métricas antigas.
"""
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from api.reconhecimento.apoio_arquivos import (
    RAIZ_API, PASTA_DADOS, PASTA_MODELOS, PASTA_RELATORIOS, ler_registro_modelo,
    calcular_resumo_arquivo, salvar_json,
)
from api.reconhecimento.apoio_montagem import montar_pipeline
from api.reconhecimento.etapa_05_classificador_integrantes import ClassificadorIntegrantes
from .etapa_01_dados import carregar_divisao


def calcular_metricas(rotulos_reais, rotulos_previstos, nomes):
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        rotulos_reais, rotulos_previstos, labels=list(range(len(nomes))), zero_division=0
    )
    return {
        'acuracia': float(accuracy_score(rotulos_reais, rotulos_previstos)),
        'precisao_macro': float(precisao.mean()), 'recall_macro': float(recall.mean()),
        'f1_macro': float(f1.mean()),
        'matriz_confusao': confusion_matrix(rotulos_reais, rotulos_previstos, labels=list(range(len(nomes)))).tolist(),
        'por_pessoa': [{'nome': nome, 'precisao': float(precisao[indice]), 'recall': float(recall[indice]),
                        'f1': float(f1[indice]), 'quantidade': int(suporte[indice])}
                       for indice, nome in enumerate(nomes)],
    }


def executar(caminho_dos_pesos=None):
    registro = ler_registro_modelo()
    _, imagens_teste = carregar_divisao('teste')  # Confere a divisão e a integridade.
    pipeline = montar_pipeline()
    pesos = PASTA_MODELOS / registro['identificacao']['arquivo']
    if caminho_dos_pesos is not None:
        pesos = Path(caminho_dos_pesos).resolve()
        pipeline.classificador = ClassificadorIntegrantes(pesos)
    previsoes, reais, previstos = [], [], []
    for imagem in imagens_teste:
        original = (RAIZ_API / imagem['original']).resolve()
        if not original.is_relative_to(PASTA_DADOS.resolve()):
            raise ValueError('Imagem original fora da pasta de dados.')
        if calcular_resumo_arquivo(original) != imagem['original_sha256']:
            raise ValueError('Foto original alterada desde o registro.')
        resultado = pipeline.identificar(original.read_bytes())
        real = registro['rotulos'].index(imagem['pessoa'])
        previsto = registro['nomes'].index(resultado['nome_previsto'])
        reais.append(real)
        previstos.append(previsto)
        previsoes.append({'arquivo': imagem['arquivo'], 'nome_real': registro['nomes'][real],
                           'nome_previsto': resultado['nome_previsto'], 'acertou': real == previsto,
                           'pontuacoes': resultado['pontuacoes'].tolist()})
    relatorio = {
        'descricao': 'Avaliação do modelo fixo em cinco fotos separadas do treino; pipeline completo desde os originais.',
        'nomes': registro['nomes'], 'quantidade_treino': 40, 'quantidade_teste': len(imagens_teste),
        'pesos_identificacao_sha256': calcular_resumo_arquivo(pesos),
        'pesos_extrator_sha256': registro['extrator']['sha256'],
        'manifesto_sha256': calcular_resumo_arquivo(PASTA_DADOS / 'manifesto.json'),
        'previsoes': previsoes, **calcular_metricas(reais, previstos, registro['nomes']),
        'limites': registro['limites'],
    }
    if caminho_dos_pesos is None:
        saida = PASTA_RELATORIOS / 'avaliacao.json'
    else:
        saida = pesos.parent / 'avaliacao.json'
    salvar_json(saida, relatorio)
    print(f'Teste: {sum(item["acertou"] for item in previsoes)}/{len(imagens_teste)} acertos. Relatório: {saida}')
    return relatorio


if __name__ == '__main__':
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument('--pesos', help='Arquivo de uma nova rede do grupo; omitido, usa o modelo da interface.')
    executar(argumentos.parse_args().pesos)
