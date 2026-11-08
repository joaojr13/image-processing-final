"""Caminhos locais e integridade dos pesos. Não importa nenhum projeto anterior."""
import hashlib
import json
from pathlib import Path

# Os caminhos do manifesto são relativos a esta pasta /api.
RAIZ_API = Path(__file__).resolve().parents[1]
PASTA_MODELOS = RAIZ_API / 'modelos'
PASTA_DADOS = RAIZ_API / 'dados'
PASTA_RELATORIOS = RAIZ_API / 'relatorios'


def calcular_resumo_arquivo(caminho):
    resumo = hashlib.sha256()
    with Path(caminho).open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            resumo.update(bloco)
    return resumo.hexdigest()


def ler_registro_modelo():
    return json.loads((PASTA_MODELOS / 'registro.json').read_text())


def conferir_pesos(registro):
    for chave in ('extrator', 'identificacao'):
        caminho = (PASTA_MODELOS / registro[chave]['arquivo']).resolve()
        if not caminho.is_relative_to(PASTA_MODELOS.resolve()):
            raise ValueError('Caminho de pesos inválido.')
        if calcular_resumo_arquivo(caminho) != registro[chave]['sha256']:
            raise ValueError('Os pesos do modelo foram alterados. Confira o registro antes de continuar.')


def salvar_json(caminho, dados):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + '.tmp')
    temporario.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + '\n')
    temporario.replace(caminho)
