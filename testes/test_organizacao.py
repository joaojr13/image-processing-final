"""Garante a separação entre apresentação e processamento após a reorganização."""
import ast
from pathlib import Path
import subprocess
import sys
import pytest
from api.etapa_01_consultas import listar_imagens_da_base, ler_codigo_explicativo

RAIZ_PROJETO = Path(__file__).resolve().parents[1]


def modulos_importados(caminho):
    arvore = ast.parse(caminho.read_text())
    for elemento in ast.walk(arvore):
        if isinstance(elemento, ast.ImportFrom) and elemento.module:
            yield elemento.module
        elif isinstance(elemento, ast.Import):
            for nome in elemento.names:
                yield nome.name


def test_backend_independente_e_telas_usam_a_camada_publica():
    for arquivo in (RAIZ_PROJETO / 'api').rglob('*.py'):
        for modulo in modulos_importados(arquivo):
            assert modulo.split('.')[0] not in ('web', 'streamlit', 'altair'), arquivo
    for arquivo in (RAIZ_PROJETO / 'web').rglob('*.py'):
        for modulo in modulos_importados(arquivo):
            if modulo.startswith('api.'):
                assert modulo in ('api.etapa_02_servicos', 'api.etapa_01_consultas'), arquivo
    subprocess.run([
        sys.executable, '-c',
        "import api.etapa_02_servicos, api.etapa_01_consultas, sys; assert 'streamlit' not in sys.modules; assert 'web' not in sys.modules",
    ], cwd=RAIZ_PROJETO, check=True, capture_output=True, timeout=30)


@pytest.mark.parametrize('divisao, quantidade', [('treino', 8), ('teste', 1)])
def test_galeria_encontra_imagens_nos_novos_caminhos(divisao, quantidade):
    imagens = listar_imagens_da_base(divisao, 'João')
    assert len(imagens) == quantidade
    assert all(item['conteudo'].startswith(b'\x89PNG') for item in imagens)


def test_consultas_recusam_caminhos_livres():
    with pytest.raises(ValueError):
        ler_codigo_explicativo('../README.md')
    with pytest.raises(ValueError):
        listar_imagens_da_base('../dados', 'João')
    with pytest.raises(ValueError):
        listar_imagens_da_base('teste', 'Pessoa não cadastrada')
    assert 'camadas.Dropout' in ler_codigo_explicativo('arquitetura')
