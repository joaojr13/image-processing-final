"""Avalia as cinco fotos originais de teste pelo pipeline completo, sem treinar.

Executar: python -m api.treinamento.etapa_05_avaliar
A avaliação registra hashes dos modelos e dados para evitar mostrar métricas antigas.
"""
import argparse
from pathlib import Path
from api.reconhecimento.apoio_arquivos import (
    PASTA_DADOS, PASTA_MODELOS, PASTA_RELATORIOS, ler_registro_modelo, localizar_imagem,
    calcular_resumo_arquivo, salvar_json,
)
from api.reconhecimento.apoio_montagem import montar_pipeline
from api.reconhecimento.etapa_05_classificador_integrantes import ClassificadorIntegrantes
from .etapa_01_dados import ler_registros_divisao
from .apoio_metricas import calcular_metricas
from api.reconhecimento.etapa_00_pipeline import LIMIAR_IDENTIFICACAO


def executar(caminho_dos_pesos=None):
    registro = ler_registro_modelo()
    imagens_teste = ler_registros_divisao('teste')
    pipeline = montar_pipeline()
    pesos = PASTA_MODELOS / registro['identificacao']['arquivo']
    if caminho_dos_pesos is not None:
        pesos = Path(caminho_dos_pesos).resolve()
        pipeline.classificador = ClassificadorIntegrantes(pesos)
    nomes_saida = registro['nomes'] + ['Desconhecido']
    previsoes = []
    reais = []
    previstos = []
    for imagem in imagens_teste:
        original = localizar_imagem(imagem['original'])
        if calcular_resumo_arquivo(original) != imagem['original_sha256']:
            raise ValueError('Foto original alterada desde o registro.')
        resultado = pipeline.identificar(original.read_bytes())
        real = registro['rotulos'].index(imagem['pessoa'])
        previsto = nomes_saida.index(resultado['nome_previsto'])
        reais.append(real)
        previstos.append(previsto)
        previsoes.append({
            'arquivo': imagem['arquivo'],
            'nome_real': registro['nomes'][real],
            'nome_previsto': resultado['nome_previsto'],
            'acertou': real == previsto,
            'pontuacoes': resultado['pontuacoes'].tolist(),
        })
    metricas = calcular_metricas(reais, previstos, registro['nomes'])
    relatorio = {
        'descricao': 'Avaliação do modelo fixo em cinco fotos separadas do treino; pipeline completo desde os originais.',
        'nomes': registro['nomes'],
        'limiar_identificacao': LIMIAR_IDENTIFICACAO,
        'quantidade_treino': registro['identificacao']['imagens_treino'],
        'quantidade_teste': len(imagens_teste),
        'pesos_identificacao_sha256': calcular_resumo_arquivo(pesos),
        'pesos_extrator_sha256': registro['extrator']['sha256'],
        'manifesto_sha256': calcular_resumo_arquivo(PASTA_DADOS / 'manifesto.json'),
        'previsoes': previsoes,
        **metricas,
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
