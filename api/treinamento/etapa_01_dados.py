"""Leitura das divisões registradas; rótulos e procedência ficam no manifesto."""
import json
from PIL import Image
import torch
from api.reconhecimento.apoio_arquivos import (
    PASTA_DADOS, calcular_resumo_arquivo, ler_registro_modelo,
    conferir_pesos, localizar_imagem,
)
from api.reconhecimento.etapa_01_preparacao import converter_para_tensor


def ler_manifesto():
    return json.loads((PASTA_DADOS / 'manifesto.json').read_text())


def conferir_separacao(registros):
    treino = [registro for registro in registros if registro['divisao'] == 'treino']
    teste = [registro for registro in registros if registro['divisao'] == 'teste']
    for campo in ('sha256', 'resumo_pixels', 'grupo_video'):
        valores_treino = {registro[campo] for registro in treino if registro.get(campo)}
        valores_teste = {registro[campo] for registro in teste if registro.get(campo)}
        if valores_treino & valores_teste:
            raise ValueError(f'Treino e teste têm informação repetida: {campo}.')


def ler_registros_divisao(divisao):
    """Seleciona a divisão e confere seus arquivos, sem transformar imagens em tensores."""
    if divisao not in ('treino', 'teste'):
        raise ValueError('Divisão esperada: treino ou teste.')
    manifesto = ler_manifesto()
    conferir_separacao(manifesto['registros'])
    registros = [registro for registro in manifesto['registros'] if registro['divisao'] == divisao]
    if not registros:
        raise ValueError(f'Não há imagens na divisão {divisao}.')
    for registro in registros:
        caminho = localizar_imagem(registro['arquivo'])
        if calcular_resumo_arquivo(caminho) != registro['sha256']:
            raise ValueError(f'Imagem alterada: {registro["arquivo"]}.')
    return registros


def carregar_divisao(divisao):
    """Converte os recortes da divisão em tensores para o treinamento."""
    registros = ler_registros_divisao(divisao)
    imagens = []
    for registro in registros:
        with Image.open(localizar_imagem(registro['arquivo'])) as imagem:
            imagens.append(converter_para_tensor(imagem))
    return torch.cat(imagens), registros


def carregar_dados_treino():
    """Confere os modelos e entrega apenas a divisão de treino ao fluxo principal."""
    registro_modelo = ler_registro_modelo()
    conferir_pesos(registro_modelo)
    imagens, registros_treino = carregar_divisao('treino')
    return registro_modelo, imagens, registros_treino
