#!/bin/bash
# Congelamento de dependências — procedimento disciplinado
# Usar Python 3.11 para consistência com runtime.txt
set -e

echo "=== 1. Ambiente limpo ==="
rm -rf venv_clean
python3.11 -m venv venv_clean || python3 -m venv venv_clean
source venv_clean/bin/activate  # Windows: venv_clean\Scripts\activate

echo "=== 2. Upgrade pip ==="
pip install --upgrade pip

echo "=== 3. Instalar pip-tools ==="
pip install pip-tools

echo "=== 4. Compilar requirements (pip-compile) ==="
pip-compile requirements.in --output-file=requirements.txt --resolver=backtracking

echo "=== 5. Concluído ==="
echo "requirements.txt atualizado com pins. Commit o arquivo."
