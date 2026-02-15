#!/usr/bin/env python3
"""
Script para iniciar a API do sistema de arbitragem
"""
import subprocess
import sys
import os
from pathlib import Path

# Mudar para o diretório do projeto
project_dir = Path(__file__).parent
os.chdir(project_dir)

print("=" * 60)
print("🚀 INICIANDO API DE ARBITRAGEM POLYMARKET")
print("=" * 60)
print()
print("📝 API: http://localhost:8000")
print("📚 Docs: http://localhost:8000/docs")
print("🔗 Análise: http://localhost:8000/polymarket/analyze")
print()
print("🛑 Pressione Ctrl+C para parar")
print("=" * 60)
print()

try:
    # Iniciar uvicorn SEM --reload (causa erro de permissão no macOS)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
except KeyboardInterrupt:
    print("\n\n✅ API encerrada")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Erro: {e}")
    sys.exit(1)

