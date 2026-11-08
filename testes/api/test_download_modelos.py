"""Downloads só entram em uso quando completos e iguais aos pesos registrados."""
import hashlib
from io import BytesIO
import pytest
from api.reconhecimento import apoio_download_modelos as modelos


def test_baixa_confere_e_reutiliza_modelo(tmp_path, monkeypatch):
    conteudo = b'pesos de teste'
    chamadas = []
    def abrir(url, timeout):
        chamadas.append(url)
        return BytesIO(conteudo)
    monkeypatch.setattr(modelos, 'urlopen', abrir)
    caminho = tmp_path / 'modelos' / 'modelo.pt'
    resumo = hashlib.sha256(conteudo).hexdigest()
    modelos.garantir_modelo_publico(caminho, 'https://exemplo.invalid/modelo', resumo)
    modelos.garantir_modelo_publico(caminho, 'https://exemplo.invalid/modelo', resumo)
    assert caminho.read_bytes() == conteudo
    assert len(chamadas) == 1
    assert not list(caminho.parent.glob('*.download'))


def test_descarta_modelo_com_integridade_incorreta(tmp_path, monkeypatch):
    monkeypatch.setattr(modelos, 'urlopen', lambda *args, **kwargs: BytesIO(b'outros pesos'))
    caminho = tmp_path / 'modelo.pt'
    with pytest.raises(ValueError, match='não corresponde'):
        modelos.garantir_modelo_publico(caminho, 'https://exemplo.invalid/modelo', 'resumo esperado')
    assert not caminho.exists()
    assert not list(tmp_path.glob('*.download'))


def test_falha_de_conexao_nao_deixa_modelo_parcial(tmp_path, monkeypatch):
    def falhar(*args, **kwargs):
        raise OSError('Conexão interrompida')
    monkeypatch.setattr(modelos, 'urlopen', falhar)
    caminho = tmp_path / 'modelo.pt'
    with pytest.raises(OSError, match='Não foi possível baixar'):
        modelos.garantir_modelo_publico(caminho, 'https://exemplo.invalid/modelo', 'resumo esperado')
    assert not caminho.exists()
    assert not list(tmp_path.glob('*.download'))
