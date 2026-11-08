"""Verificação visual dos resultados de expressão."""
from streamlit.testing.v1 import AppTest


def test_interface_exibe_expressao_e_indisponibilidade():
    pagina = AppTest.from_string('''
from PIL import Image
import numpy as np
from web.telas.etapa_03_expressoes import mostrar_expressao
from api.expressoes.etapa_02_analisador_ferplus import NOMES_EXPRESSOES
mostrar_expressao({'expressao': {
    'nome': 'Alegria', 'pontuacao': 0.65, 'nomes': list(NOMES_EXPRESSOES),
    'pontuacoes': np.array([0.05, 0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    'imagem_cinza': Image.new('L', (64, 64)),
}})
mostrar_expressao({'erro_expressao': 'Expressão indisponível; identificação preservada.'})
''').run()
    assert not pagina.exception
    assert pagina.subheader[0].value == 'Alegria'
    assert 'indisponível' in pagina.info[0].value
    assert any('Não determina' in legenda.value for legenda in pagina.caption)
