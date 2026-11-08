"""ARQUITETURA COMPLETA DA NOSSA REDE EM KERAS, sem incluir a CNN externa.

Entrada: 512 características do rosto, fornecidas pela FaceNet congelada.
Camadas: 1 Dropout + 1 Dense. Apenas Dense possui parâmetros treináveis.
Saída: 5 valores brutos, na ordem definida no registro do modelo.
Total aprendido pelo grupo: 512 × 5 pesos + 5 vieses = 2.565 parâmetros.
Esta é uma rede classificadora sem convoluções, não outra CNN treinada do zero.
"""
from api.apoio_configuracao_keras import keras

camadas = keras.layers


class RedeIdentificacaoPessoas(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                # ENTRADA — define o formato; não acrescenta uma camada treinável.
                keras.Input(shape=(512,), name='caracteristicas_do_rosto'),

                # CAMADA 1 — desativa 10% das características somente no treino.
                camadas.Dropout(rate=0.1, name='dropout'),

                # CAMADA 2 — aprende 512 × 5 pesos e 5 vieses: 2.565 parâmetros.
                camadas.Dense(units=5, activation=None, name='camada_de_saida'),
            ],
            name='rede_identificacao_pessoas',
        )
        # Sequential executa as camadas na ordem acima; não precisamos de forward.
        # Softmax é aplicado depois da rede, em etapa_00_pipeline.py.
