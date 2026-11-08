"""FER+ pronto: estima expressão; não reconhece identidade e não é treinado aqui.

Fonte: ONNX Model Zoo, emotion-ferplus-8. Execução local, sem enviar fotos.
A arquitetura interna pertence ao modelo externo, armazenado em ferplus.onnx.
"""
import json
from api.reconhecimento.apoio_arquivos import PASTA_MODELOS, calcular_resumo_arquivo
from .apoio_contratos import AnalisadorExpressao, ErroAnaliseExpressao
from .etapa_01_preparacao import preparar_rosto_para_expressao, converter_saidas_em_pontuacoes

# Ordem OFICIAL das oito saídas da rede. Não pode ser reorganizada.
NOMES_EXPRESSOES = ('Neutra', 'Alegria', 'Surpresa', 'Tristeza', 'Raiva', 'Nojo', 'Medo', 'Desprezo')
PASTA_EXPRESSOES = PASTA_MODELOS / 'expressoes'


class AnalisadorFERPlus(AnalisadorExpressao):
    def __init__(self, pasta_modelo=PASTA_EXPRESSOES):
        self.pasta_modelo = pasta_modelo
        self.sessao = None

    def carregar_rede_pronta(self):
        """Carrega uma vez. Nenhum peso da identificação depende deste arquivo."""
        try:
            import onnxruntime as executor_onnx
            registro = json.loads((self.pasta_modelo / 'registro.json').read_text())
            caminho = self.pasta_modelo / 'ferplus.onnx'
            if calcular_resumo_arquivo(caminho) != registro['sha256']:
                raise ValueError('Os pesos de expressão diferem do registro.')
            if list(NOMES_EXPRESSOES) != registro['classes']:
                raise ValueError('A ordem das expressões difere da ordem do modelo.')
            opcoes = executor_onnx.SessionOptions()
            opcoes.intra_op_num_threads = 2
            opcoes.inter_op_num_threads = 1
            self.sessao = executor_onnx.InferenceSession(
                str(caminho), sess_options=opcoes, providers=['CPUExecutionProvider']
            )
            entrada = self.sessao.get_inputs()[0]
            if entrada.shape != [1, 1, 64, 64] or entrada.type != 'tensor(float)':
                raise ValueError('A entrada do modelo FER+ é incompatível.')
            self.nome_entrada = entrada.name
        except Exception as erro:
            # Limite com a biblioteca externa: qualquer falha aqui afeta só expressão.
            self.sessao = None
            raise ErroAnaliseExpressao(
                'Análise de expressão indisponível. Confira a instalação do ONNX Runtime '
                'e os arquivos em api/modelos/expressoes. A identificação continua disponível.'
            ) from erro

    def analisar(self, rosto_alinhado):
        if self.sessao is None:
            self.carregar_rede_pronta()
        try:
            imagem_cinza, pixels = preparar_rosto_para_expressao(rosto_alinhado)
            saidas = self.sessao.run(None, {self.nome_entrada: pixels})[0]
            pontuacoes = converter_saidas_em_pontuacoes(saidas)
        except Exception as erro:
            raise ErroAnaliseExpressao('Não foi possível analisar a expressão desta foto.') from erro
        indice = int(pontuacoes.argmax())
        return {
            'nome': NOMES_EXPRESSOES[indice],
            'pontuacao': float(pontuacoes[indice]),
            'nomes': list(NOMES_EXPRESSOES),
            'pontuacoes': pontuacoes,
            'imagem_cinza': imagem_cinza,
        }
