"""Fluxo completo do treino. Cada chamada abaixo corresponde a um arquivo de etapa.

Executar: python -m api.treinamento.etapa_00_treinar
O modelo ativo da interface não é substituído.
"""
from .etapa_01_dados import carregar_dados_treino
from .etapa_02_extrair_caracteristicas import extrair_caracteristicas
from .etapa_03_aprender import treinar_rede
from .etapa_04_salvar import salvar_treinamento


def executar():
    # 1. Carregar somente as imagens reservadas para treino.
    registro_modelo, imagens, registros_treino = carregar_dados_treino()

    # 2. Extrair as características com a FaceNet congelada.
    caracteristicas, rotulos = extrair_caracteristicas(imagens, registros_treino, registro_modelo)

    # 3. Treinar somente a nossa rede Keras.
    rede, curvas = treinar_rede(caracteristicas, rotulos)

    # 4. Salvar os novos pesos e o histórico do treino.
    caminho = salvar_treinamento(rede, curvas, registro_modelo, registros_treino)
    print(f'Treinamento concluído. Pesos salvos em: {caminho}')
    print(f'Para avaliar: python -m api.treinamento.etapa_05_avaliar --pesos {caminho}')
    return caminho


if __name__ == '__main__':
    executar()
