"""Verificações da separação das redes, do preparo e da equivalência do modelo."""
from io import BytesIO
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from api.reconhecimento.apoio_contratos import DetectorRosto, ExtratorCaracteristicas, ClassificadorIdentidade, DeteccaoRosto
from api.reconhecimento.etapa_01_preparacao import (
    carregar_imagem, corrigir_orientacao, converter_para_tensor, redimensionar_rosto,
    normalizar_pixels, calcular_matriz_alinhamento, OLHOS_DESTINO,
)
from api.reconhecimento.etapa_04_rede_identificacao_pessoas import RedeIdentificacaoPessoas
from api.reconhecimento.etapa_03_extrator_facenet import RedeExternaCongelada
from api.reconhecimento.etapa_00_pipeline import PipelineReconhecimento
from api.reconhecimento.apoio_arquivos import ler_registro_modelo
from api.treinamento.etapa_03_aprender import treinar_rede
from api.treinamento.etapa_01_dados import conferir_separacao
from api.etapa_01_consultas import carregar_avaliacao
from api.apoio_configuracao_keras import keras


def test_arquitetura_completa_e_dropout_inativo_na_avaliacao():
    rede = RedeIdentificacaoPessoas()
    assert isinstance(rede, keras.Sequential)
    assert [camada.name for camada in rede.layers] == ['dropout', 'camada_de_saida']
    assert rede.count_params() == 2565
    assert rede.input_shape == (None, 512)
    assert rede.output_shape == (None, 5)
    entrada = np.ones((2, 512), dtype=np.float32)
    esperado = keras.ops.convert_to_numpy(rede.get_layer('camada_de_saida')(entrada))
    np.testing.assert_allclose(rede.predict(entrada, verbose=0), esperado, atol=1e-6)
    dropout = rede.get_layer('dropout')
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(dropout(entrada, training=False)), entrada)
    assert (keras.ops.convert_to_numpy(dropout(entrada, training=True)) == 0).any()
    with pytest.raises(ValueError):
        rede(np.zeros((1, 3, 96, 96), dtype=np.float32))


def test_extrator_nao_recebe_gradientes_nem_atualiza_batchnorm():
    base = nn.Sequential(nn.Linear(3,512), nn.BatchNorm1d(512), nn.Dropout(0.5))
    extrator = RedeExternaCongelada(base)
    estado_anterior = {k:v.clone() for k,v in base.state_dict().items()}
    extrator.train()
    caracteristicas = extrator(torch.randn(10,3,requires_grad=True))
    keras.utils.set_random_seed(42)
    pesos_iniciais = RedeIdentificacaoPessoas().get_weights()
    rede, curvas = treinar_rede(caracteristicas.numpy(), np.tile(np.arange(5), 2), epocas=2)
    assert not extrator.training and not base.training
    assert not caracteristicas.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(torch.equal(v,estado_anterior[k]) for k,v in base.state_dict().items())
    assert len(curvas) == 2 and np.isfinite([item['perda_treino'] for item in curvas]).all()
    assert any(not np.array_equal(antes, depois) for antes, depois in zip(pesos_iniciais, rede.get_weights()))


def test_normalizacao_e_alinhamento_geometrico():
    preto = Image.new('RGB',(96,96),'black')
    branco = Image.new('RGB',(96,96),'white')
    lote = torch.cat([converter_para_tensor(preto),converter_para_tensor(branco)])
    saida = normalizar_pixels(redimensionar_rosto(lote))
    assert saida.shape==(2,3,160,160)
    assert torch.all(saida[0]==-127.5/128)
    assert torch.all(saida[1]==127.5/128)
    olhos=np.array([[80.,90.],[160.,110.]])
    matriz=calcular_matriz_alinhamento(olhos)
    calculados=(matriz@np.column_stack([olhos,np.ones(2)]).T).T[:,:2]
    np.testing.assert_allclose(calculados,OLHOS_DESTINO,atol=1e-10)


def test_arquivo_invalido_limites_e_orientacao():
    with pytest.raises(ValueError,match='abrir'):
        carregar_imagem(b'nao e uma foto')
    with pytest.raises(ValueError,match='12 MB'):
        carregar_imagem(b'x'*(12*1024*1024+1))
    imagem=Image.new('RGB',(40,20),'red')
    exif=imagem.getexif();exif[274]=6
    arquivo=BytesIO();imagem.save(arquivo,format='JPEG',exif=exif)
    assert corrigir_orientacao(carregar_imagem(arquivo.getvalue())).size==(20,40)


def test_contratos_e_ordem_do_pipeline():
    passos=[]
    class DetectorDeTeste(DetectorRosto):
        def detectar(self,imagem):
            passos.append('detectar')
            return DeteccaoRosto(np.array([5,5,90,90]),np.array([[30.72,36.48],[65.28,36.48],[48,50],[35,65],[60,65]]),1.)
    class ExtratorDeTeste(ExtratorCaracteristicas):
        def extrair(self,pixels):
            passos.append('extrair');assert pixels.shape==(1,3,160,160)
            return np.zeros((1,512), dtype=np.float32)
    class ClassificadorDeTeste(ClassificadorIdentidade):
        def classificar(self,caracteristicas):
            passos.append('classificar');assert caracteristicas.shape==(1,512)
            return np.array([[0.,0.,0.,10.,0.]], dtype=np.float32)
    arquivo=BytesIO();Image.new('RGB',(96,96),'white').save(arquivo,format='PNG')
    pipeline=PipelineReconhecimento(DetectorDeTeste(),ExtratorDeTeste(),ClassificadorDeTeste(),['Diego','Fernanda','Guilherme','João','Samanta'])
    resultado=pipeline.identificar(arquivo.getvalue())
    assert passos==['detectar','extrair','classificar']
    assert resultado['nome_previsto']=='João'
    assert abs(resultado['pontuacoes'].sum()-1)<1e-6


def test_proibe_repeticao_de_pixels_e_video_entre_divisoes():
    registros=[{'divisao':'treino','sha256':'a','resumo_pixels':'x','grupo_video':'v'},
               {'divisao':'teste','sha256':'b','resumo_pixels':'y','grupo_video':None}]
    conferir_separacao(registros)
    registros[1]['grupo_video']='v'
    with pytest.raises(ValueError,match='grupo_video'):
        conferir_separacao(registros)
    registros[1]['grupo_video']=None;registros[1]['resumo_pixels']='x'
    with pytest.raises(ValueError,match='resumo_pixels'):
        conferir_separacao(registros)


def test_relatorio_compativel_com_modelo_e_previsoes():
    registro=ler_registro_modelo()
    relatorio=carregar_avaliacao(registro)
    assert relatorio is not None
    assert relatorio['quantidade_teste']==5
    assert sum(map(sum,relatorio['matriz_confusao']))==5
    acertos = sum(item['nome_real'] == item['nome_previsto'] for item in relatorio['previsoes'])
    assert relatorio['acuracia'] == acertos / relatorio['quantidade_teste']
    registro['identificacao']['sha256']='alterado'
    assert carregar_avaliacao(registro) is None


@pytest.mark.parametrize('pontuacoes, mensagem', [
    (None, 'Nenhum rosto'),
    ([0.80], 'Nenhum rosto'),
    ([0.99, 0.98], 'mais de um rosto'),
])
def test_detector_recusa_ausencia_baixa_pontuacao_e_multiplos_rostos(pontuacoes, mensagem):
    from api.reconhecimento.etapa_02_detector_mtcnn import DetectorMTCNN

    class RedeDetectoraDeTeste:
        def detect(self, imagem, landmarks):
            if pontuacoes is None:
                return None, None, None
            quantidade = len(pontuacoes)
            return np.zeros((quantidade, 4)), pontuacoes, np.zeros((quantidade, 5, 2))

    detector = DetectorMTCNN.__new__(DetectorMTCNN)
    detector.rede_detectora = RedeDetectoraDeTeste()
    with pytest.raises(ValueError, match=mensagem):
        detector.detectar(Image.new('RGB', (96, 96)))
