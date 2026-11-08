"""Operações do backend usadas pela interface, com entradas simples em Python.

A interface envia os bytes de uma foto. O pipeline entrega imagens intermediárias,
nome, expressão e pontuações; desenhar e formatar esses resultados cabe a /web.
"""
from .reconhecimento.apoio_montagem import montar_pipeline
from .reconhecimento.etapa_01_preparacao import carregar_imagem, corrigir_orientacao, converter_para_rgb


def criar_analisador_fotos():
    """Prepara as redes de identidade e expressão para reutilizar entre fotos."""
    return montar_pipeline(incluir_expressao=True)


def preparar_previa_foto(conteudo):
    """Valida e orienta a imagem enviada antes de ela ser exibida na interface."""
    imagem = carregar_imagem(conteudo)
    imagem_orientada = corrigir_orientacao(imagem)
    return converter_para_rgb(imagem_orientada)
