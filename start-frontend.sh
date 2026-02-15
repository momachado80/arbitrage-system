#!/bin/bash

# Script para iniciar o servidor frontend

echo "🚀 Iniciando servidor frontend..."
cd "$(dirname "$0")/frontend"

# Verificar se a porta 8080 está em uso
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Porta 8080 já está em uso"
    echo "📝 Acesse: http://localhost:8080/polymarket-analyzer.html"
else
    echo "✅ Servidor iniciado na porta 8080"
    echo "📝 Acesse: http://localhost:8080/polymarket-analyzer.html"
    echo ""
    echo "⚠️  IMPORTANTE: Certifique-se de que a API está rodando em http://localhost:8000"
    echo "   Para iniciar a API: uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    python3 -m http.server 8080
fi




