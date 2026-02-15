#!/usr/bin/env python3
"""
Script para iniciar o servidor HTTP do frontend
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

# Mudar para o diretório frontend
frontend_dir = Path(__file__).parent / "frontend"
os.chdir(frontend_dir)

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Adicionar headers CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"🚀 Servidor iniciado!")
            print(f"📝 Acesse: http://localhost:{PORT}/polymarket-analyzer.html")
            print(f"📁 Servindo arquivos de: {frontend_dir}")
            print(f"\n⚠️  IMPORTANTE: Certifique-se de que a API está rodando em http://localhost:8000")
            print(f"   Para iniciar a API: uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
            print(f"\n🛑 Pressione Ctrl+C para parar o servidor\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Servidor encerrado")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️  Porta {PORT} já está em uso")
            print(f"📝 Acesse: http://localhost:{PORT}/polymarket-analyzer.html")
        else:
            print(f"❌ Erro: {e}")
            sys.exit(1)




