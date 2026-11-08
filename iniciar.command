#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Ambiente não encontrado. Consulte a instalação no README.md."
  exit 1
fi
exec .venv/bin/python -m streamlit run web/etapa_01_app.py
