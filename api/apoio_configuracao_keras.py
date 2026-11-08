"""Configura Keras 3 antes do primeiro import, sem alterar preferências globais.

Keras define e treina nossa rede. PyTorch é o motor numérico do Keras e também
executa as redes externas MTCNN e FaceNet. A configuração vale neste processo.
"""
import os
from .reconhecimento.apoio_arquivos import PASTA_MODELOS

os.environ['KERAS_BACKEND'] = 'torch'
os.environ.setdefault('KERAS_HOME', str(PASTA_MODELOS / '.keras'))

import keras

if keras.backend.backend() != 'torch':
    raise RuntimeError('Keras já foi carregado com outro motor. Reinicie a aplicação para usar PyTorch.')
