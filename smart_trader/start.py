#!/usr/bin/env python3
"""
Smart Event Trader - Script de Inicialização
=============================================
Inicia a API e o frontend.
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# Diretório base
BASE_DIR = Path(__file__).parent

def print_banner():
    """Imprime banner de inicialização"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎯 SMART EVENT TRADER                                   ║
║   ──────────────────────────────────────────────────      ║
║   Sistema de Detecção de Eventos + Sinais de Trading      ║
║                                                           ║
║   📰 Monitora: Breaking News (NewsAPI, RSS, Google)       ║
║   🐋 Monitora: Whale Movement (Polygon/Etherscan)         ║
║   📊 Monitora: Polymarket Price Action                    ║
║   ⏱️ Calcula: Correlação Temporal (Insider Detection)     ║
║                                                           ║
║   Alertas via: Telegram, Push, Email                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Verifica dependências necessárias"""
    required = ["fastapi", "uvicorn", "aiohttp", "pydantic", "websockets"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️ Instalando dependências: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing, "-q"])
    
    return True

def start_api():
    """Inicia a API em background"""
    print("🚀 Iniciando API...")
    
    api_path = BASE_DIR / "backend" / "api" / "main.py"
    
    # Adiciona o diretório ao PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR / "backend")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", 
         "--host", "0.0.0.0", "--port", "8001", "--reload"],
        cwd=str(BASE_DIR / "backend"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process

def start_frontend():
    """Inicia o servidor do frontend"""
    print("🌐 Iniciando Frontend...")
    
    frontend_path = BASE_DIR / "frontend"
    
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8002"],
        cwd=str(frontend_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process

def main():
    """Função principal"""
    print_banner()
    
    # Verifica dependências
    check_dependencies()
    
    # Inicia serviços
    api_process = start_api()
    frontend_process = start_frontend()
    
    # Aguarda inicialização
    print("\n⏳ Aguardando inicialização dos serviços...")
    time.sleep(3)
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║  ✅ SMART EVENT TRADER INICIADO!                          ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  🌐 Dashboard:  http://localhost:8002                     ║
║  📡 API:        http://localhost:8001                     ║
║  📚 API Docs:   http://localhost:8001/docs                ║
║  🔌 WebSocket:  ws://localhost:8001/ws                    ║
║                                                           ║
║  ──────────────────────────────────────────────────       ║
║                                                           ║
║  📝 Configuração de Alertas (variáveis de ambiente):      ║
║                                                           ║
║  NEWSAPI_KEY=sua_chave          # Para news               ║
║  POLYGONSCAN_KEY=sua_chave      # Para whale tracking     ║
║  TELEGRAM_BOT_TOKEN=token       # Para alertas Telegram   ║
║  TELEGRAM_CHAT_ID=chat_id       # Chat ID do Telegram     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

💡 Pressione Ctrl+C para encerrar.
""")
    
    # Abre navegador
    try:
        webbrowser.open("http://localhost:8002")
    except:
        pass
    
    # Mantém rodando
    try:
        while True:
            time.sleep(1)
            
            # Verifica se processos estão rodando
            if api_process.poll() is not None:
                print("⚠️ API parou. Reiniciando...")
                api_process = start_api()
            
            if frontend_process.poll() is not None:
                print("⚠️ Frontend parou. Reiniciando...")
                frontend_process = start_frontend()
    
    except KeyboardInterrupt:
        print("\n🛑 Encerrando Smart Event Trader...")
        api_process.terminate()
        frontend_process.terminate()
        print("👋 Até mais!")

if __name__ == "__main__":
    main()

