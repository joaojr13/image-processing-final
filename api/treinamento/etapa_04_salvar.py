"""Salva os pesos e o registro de um novo treino, sem substituir o modelo ativo."""
from datetime import datetime, timezone
from api.apoio_configuracao_keras import keras
from api.reconhecimento.apoio_arquivos import PASTA_MODELOS, salvar_json, calcular_resumo_arquivo
from .apoio_parametros import SEMENTE, TAXA_APRENDIZADO, DECAIMENTO_PESOS, EPSILON


def salvar_treinamento(rede, curvas, registro_modelo, registros_treino, semente=SEMENTE):
    """Cria uma pasta com data/hora, grava os pesos e documenta quais dados foram usados."""
    identificador = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    pasta = PASTA_MODELOS / 'treinamentos' / identificador
    pasta.mkdir(parents=True, exist_ok=False)
    caminho = pasta / 'identificacao_pessoas.weights.h5'
    rede.save_weights(caminho)

    arquivos_treino = []
    for imagem in registros_treino:
        arquivos_treino.append(imagem['arquivo'])

    registro_treinamento = {
        'nomes': registro_modelo['nomes'],
        'imagens_treino': len(registros_treino),
        'biblioteca': 'Keras',
        'versao_keras': keras.__version__,
        'motor': keras.backend.backend(),
        'arquitetura': 'Dropout(0.1) → Dense(5), entrada 512',
        'parametros_treinaveis': rede.count_params(),
        'epocas': len(curvas),
        'semente': semente,
        'otimizador': 'AdamW',
        'taxa_aprendizado': TAXA_APRENDIZADO,
        'decaimento_pesos': DECAIMENTO_PESOS,
        'epsilon': EPSILON,
        'curvas': curvas,
        'pesos_sha256': calcular_resumo_arquivo(caminho),
        'extrator_sha256': registro_modelo['extrator']['sha256'],
        'arquivos_treino': arquivos_treino,
    }
    salvar_json(pasta / 'registro.json', registro_treinamento)
    return caminho
