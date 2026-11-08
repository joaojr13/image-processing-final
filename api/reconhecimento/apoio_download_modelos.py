"""Obtém pesos públicos ausentes e verifica sua integridade antes de utilizá-los."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
import shutil

URL_FACENET = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'


def garantir_modelo_publico(caminho, url, resumo_esperado):
    """Reutiliza o arquivo local ou baixa uma cópia verificada da fonte registrada."""
    caminho = Path(caminho)
    if caminho.exists():
        return  # O carregador confere os pesos locais antes de usá-los.
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = None
    try:
        # Um arquivo incompleto nunca ocupa o caminho definitivo do modelo.
        with NamedTemporaryFile(dir=caminho.parent, suffix='.download', delete=False) as arquivo:
            temporario = Path(arquivo.name)
            with urlopen(url, timeout=60) as resposta:
                shutil.copyfileobj(resposta, arquivo)
        with temporario.open('rb') as arquivo:
            resumo = hashlib.file_digest(arquivo, 'sha256').hexdigest()
        if resumo != resumo_esperado:
            raise ValueError('O modelo baixado não corresponde aos pesos registrados.')
        temporario.replace(caminho)
    except OSError as erro:
        raise OSError(f'Não foi possível baixar {caminho.name}. Verifique a conexão e tente novamente.') from erro
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)
