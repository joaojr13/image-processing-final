"""Aqui nossa rede aprende: compile() define o treino e fit() ajusta os pesos."""
import numpy as np
from api.apoio_configuracao_keras import keras
from api.reconhecimento.etapa_04_rede_identificacao_pessoas import RedeIdentificacaoPessoas
from .apoio_parametros import EPOCAS, SEMENTE, TAXA_APRENDIZADO, DECAIMENTO_PESOS, EPSILON
from .apoio_validacao_treino import validar_entradas, calcular_pesos_por_pessoa


def treinar_rede(caracteristicas_treino, rotulos_treino, epocas=EPOCAS, semente=SEMENTE):
    """Recebe somente dados de treino. As fotos de teste não entram nesta função."""
    caracteristicas = np.asarray(caracteristicas_treino, dtype=np.float32)
    rotulos = np.asarray(rotulos_treino)
    validar_entradas(caracteristicas, rotulos, epocas)

    keras.utils.set_random_seed(semente)
    rede = RedeIdentificacaoPessoas()
    pesos_por_pessoa = calcular_pesos_por_pessoa(rotulos)

    # 1. Definir como medir o erro e como ajustar a nossa rede.
    rede.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=TAXA_APRENDIZADO, weight_decay=DECAIMENTO_PESOS, epsilon=EPSILON,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=False,
    )

    # 2. Aprender. Keras ativa Dropout e atualiza os 2.565 parâmetros da nossa rede.
    historico = rede.fit(
        caracteristicas, rotulos,
        epochs=epocas, batch_size=len(rotulos), shuffle=False,
        class_weight=pesos_por_pessoa, verbose=0,
    )

    # 3. Registrar o erro de cada época para acompanhar o aprendizado.
    curvas = []
    for epoca, perda in enumerate(historico.history['loss'], start=1):
        curvas.append({'epoca': epoca, 'perda_treino': float(perda)})
    return rede, curvas
