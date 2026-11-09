"""Consultas para a interface: configuração, avaliação, galeria e código didático.

Este módulo lê arquivos do backend e devolve dados. Não cria elementos de tela.
"""
import json
from .reconhecimento.apoio_arquivos import (
    RAIZ_API, PASTA_DADOS, PASTA_MODELOS, PASTA_RELATORIOS,
    calcular_resumo_arquivo, ler_registro_modelo, localizar_imagem,
)


def carregar_contexto_modelos():
    """Registro de nomes e chave que invalida o cache quando os modelos mudam."""
    registro = ler_registro_modelo()
    identificador = calcular_resumo_arquivo(PASTA_MODELOS / 'registro.json')
    registro_expressao = PASTA_MODELOS / 'expressoes' / 'registro.json'
    if registro_expressao.exists():
        identificador += calcular_resumo_arquivo(registro_expressao)
    return registro, identificador


def carregar_avaliacao(registro):
    """Entrega métricas apenas quando correspondem aos modelos e dados atuais."""
    caminho = PASTA_RELATORIOS / 'avaliacao.json'
    if not caminho.exists():
        return None
    relatorio = json.loads(caminho.read_text())
    confere = (
        relatorio['pesos_identificacao_sha256'] == registro['identificacao']['sha256']
        and relatorio['pesos_extrator_sha256'] == registro['extrator']['sha256']
        and relatorio['manifesto_sha256'] == calcular_resumo_arquivo(PASTA_DADOS / 'manifesto.json')
    )
    return relatorio if confere else None


def listar_imagens_da_base(divisao, nome):
    """Entrega as imagens de uma pessoa e divisão, sem expor caminhos à tela."""
    if divisao not in ('treino', 'teste'):
        raise ValueError('Divisão esperada: treino ou teste.')
    registro = ler_registro_modelo()
    if nome not in registro['nomes']:
        raise ValueError('Integrante não cadastrado.')
    rotulo = registro['rotulos'][registro['nomes'].index(nome)]
    manifesto = json.loads((PASTA_DADOS / 'manifesto.json').read_text())
    imagens = []
    for item in manifesto['registros']:
        if item['divisao'] != divisao or item['pessoa'] != rotulo:
            continue
        caminho = localizar_imagem(item['arquivo'])
        imagens.append({'conteudo': caminho.read_bytes(), 'tipo': item['tipo']})
    return imagens


def ler_codigo_explicativo(etapa):
    """Lê o código real de uma das etapas didáticas conhecidas."""
    arquivos = {
        'arquitetura': 'reconhecimento/etapa_04_rede_identificacao_pessoas.py',
        'pipeline': 'reconhecimento/etapa_00_pipeline.py',
        'treinamento': 'treinamento/etapa_00_treinar.py',
        'aprendizado': 'treinamento/etapa_03_aprender.py',
        'preparacao_expressao': 'expressoes/etapa_01_preparacao.py',
        'modelo_expressao': 'expressoes/etapa_02_analisador_ferplus.py',
    }
    if etapa not in arquivos:
        raise ValueError('Etapa explicativa desconhecida.')
    return (RAIZ_API / arquivos[etapa]).read_text()
