# Como Acessar o Sistema de Arbitragem

Este documento explica todas as formas de acessar e usar o sistema de análise de arbitragem.

## 📋 Índice

1. [Linha de Comando (Script Python)](#1-linha-de-comando-script-python)
2. [API REST (FastAPI)](#2-api-rest-fastapi)
3. [Interface Web](#3-interface-web)
4. [Como Módulo Python](#4-como-módulo-python)

---

## 1. Linha de Comando (Script Python)

### Análise de Arbitragem do Polymarket

Execute diretamente o script de análise:

```bash
python analyze_polymarket.py
```

**O que faz:**
- Escaneia mercados do Polymarket em tempo real
- Filtra por volume > $100.000
- Identifica divergências de probabilidade
- Apresenta Top 3 oportunidades com maior EV positivo

**Saída:**
- Formatação visual no terminal
- Tabelas de retornos projetados
- Estratégias recomendadas

---

## 2. API REST (FastAPI)

### Iniciar o Servidor

```bash
# Opção 1: Usando uvicorn diretamente
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Opção 2: Executar api.py diretamente
python api.py
```

O servidor estará disponível em: **http://localhost:8000**

### Documentação Interativa

Acesse a documentação automática da API:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Principais

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Análise de Arbitragem Polymarket (NOVO)
```bash
# GET - Análise completa de oportunidades
curl http://localhost:8000/polymarket/analyze?limit=200

# Com parâmetros
curl "http://localhost:8000/polymarket/analyze?limit=100"
```

**Resposta JSON:**
```json
{
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00",
  "total_markets_scanned": 200,
  "opportunities_found": 3,
  "opportunities": [
    {
      "market_title": "Will Bitcoin reach $100k?",
      "condition_id": "0x1234...",
      "polymarket_price": 0.65,
      "implied_probability": 0.65,
      "expected_probability": 0.75,
      "divergence": 0.10,
      "volume_24h": 150000.0,
      "liquidity_depth": 50000.0,
      "time_to_resolution_hours": 24.5,
      "expected_value": 0.10,
      "edge_percentage": 10.0,
      "recommended_side": "YES",
      "recommended_investment": 500.0,
      "returns_10": 1.0,
      "returns_20": 2.0,
      "returns_100": 10.0,
      "returns_1000": 100.0,
      "confidence_score": 0.75,
      "detected_at": "2024-01-15T10:30:00"
    }
  ],
  "criteria": {
    "min_volume_usd": 100000.0,
    "max_timeframe_hours": 48,
    "min_divergence": 0.10,
    "min_ev": 0.05
  }
}
```

#### Outros Endpoints Disponíveis

```bash
# Listar mercados Kalshi
curl http://localhost:8000/markets/kalshi?limit=10

# Listar mercados Polymarket
curl http://localhost:8000/markets/polymarket?limit=10

# Analisar par específico
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "kalshi_ticker": "TICKER",
    "polymarket_condition_id": "0x1234...",
    "target_usd": 20.0
  }'

# Scan automático
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "target_usd": 20.0,
    "min_similarity": 0.3,
    "max_markets": 30
  }'
```

---

## 3. Interface Web

O projeto inclui interfaces web HTML na pasta `frontend/`:

### Abrir no Navegador

```bash
# Navegue até a pasta frontend
cd frontend

# Abra os arquivos HTML no navegador:
# - index.html - Interface principal
# - scanner.html - Scanner de arbitragem
# - full-scanner.html - Scanner completo
```

**Nota:** As interfaces web precisam que a API esteja rodando em `http://localhost:8000`

### Usando Python HTTP Server

```bash
# Na pasta frontend
python -m http.server 8080

# Acesse: http://localhost:8080/index.html
```

---

## 4. Como Módulo Python

### Análise de Arbitragem Polymarket

```python
import asyncio
from src.engine.arbitrage_analyst import ArbitrageAnalyst

async def main():
    async with ArbitrageAnalyst() as analyst:
        # Escanear oportunidades
        opportunities = await analyst.scan_opportunities(limit=200)
        
        # Top 3
        top_3 = opportunities[:3]
        
        for i, opp in enumerate(top_3, 1):
            print(f"\n=== Oportunidade #{i} ===")
            print(f"Mercado: {opp.market_title}")
            print(f"Preço: ${opp.polymarket_price:.4f}")
            print(f"Edge: {opp.edge_percentage:.2f}%")
            print(f"EV: ${opp.expected_value:.4f} por $1")
            print(f"Retorno $100: ${opp.returns_100:.2f}")
            print(f"Lado: {opp.recommended_side.value}")
        
        # Formatar saída completa
        summary = analyst.format_summary(opportunities)
        print(summary)

if __name__ == "__main__":
    asyncio.run(main())
```

### Scanner de Arbitragem Kalshi-Polymarket

```python
import asyncio
from decimal import Decimal
from src.main import ArbitrageScanner

async def main():
    scanner = ArbitrageScanner(target_usd=Decimal("20"))
    
    # Adicionar pares de mercados
    scanner.add_market_pair(
        kalshi_ticker="KALSHI_TICKER",
        polymarket_condition_id="0x1234...",
        equivalence_score=Decimal("1.0")
    )
    
    # Executar scan
    opportunities = await scanner.scan()
    
    print(f"Encontradas {len(opportunities)} oportunidades")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🚀 Deploy em Produção

### Railway

O projeto está configurado para deploy no Railway:

```bash
# O Railway detecta automaticamente:
# - Procfile: web: uvicorn api:app --host 0.0.0.0 --port $PORT
# - requirements.txt: dependências Python
# - railway.json: configurações de deploy
```

### Variáveis de Ambiente

Configure no Railway (se necessário):
- `ANTHROPIC_API_KEY`: Para análise semântica avançada
- `PORT`: Porta do servidor (gerenciado automaticamente)

---

## 📊 Resumo das Formas de Acesso

| Método | Comando | Uso |
|--------|---------|-----|
| **Script CLI** | `python analyze_polymarket.py` | Análise rápida no terminal |
| **API Local** | `uvicorn api:app --reload` | Acesso via HTTP/JSON |
| **API Docs** | http://localhost:8000/docs | Documentação interativa |
| **Interface Web** | Abrir `frontend/index.html` | Interface visual |
| **Módulo Python** | `from src.engine.arbitrage_analyst import ...` | Integração em código |

---

## 🔧 Troubleshooting

### Erro: "Module not found"
```bash
# Certifique-se de estar no diretório raiz do projeto
cd /Users/momachado/Desktop/arbitrage-system

# Instale as dependências
pip install -r requirements.txt
```

### Erro: "Port already in use"
```bash
# Use outra porta
uvicorn api:app --port 8001
```

### Erro de conexão com APIs
- Verifique sua conexão com a internet
- As APIs do Polymarket e Kalshi são públicas, mas podem ter rate limits

---

## 5. 🐋 Whale Tracker - Rastreador de Insiders

### Acesso via Interface Web

```
http://localhost:8080/whale-tracker.html
```

### O que é?

O Whale Tracker é um sistema que monitora carteiras no Polymarket para detectar:

- **Whales**: Carteiras com grande volume de negociação (>$10.000)
- **Carteiras Novas**: Carteiras criadas recentemente fazendo trades grandes
- **Timing Suspeito**: Trades grandes feitos pouco antes da resolução de eventos
- **Posições Concentradas**: Carteiras focadas em poucos mercados
- **Trading Coordenado**: Múltiplas carteiras agindo em sincronia

### Endpoints da API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/whale/scan` | POST | Escaneia mercados em busca de whales |
| `/whale/analyze-market` | POST | Analisa atividade de um mercado específico |
| `/whale/watch` | POST | Adiciona carteira ao monitoramento |
| `/whale/alerts` | GET | Retorna alertas de atividade suspeita |
| `/whale/stats` | GET | Estatísticas do sistema |

### Exemplo de Uso

```bash
# Escanear por whales (carteiras com >$5000 de volume)
curl -X POST "http://localhost:8000/whale/scan" \
  -H "Content-Type: application/json" \
  -d '{"min_volume_usd": 5000, "lookback_hours": 24}'

# Monitorar uma carteira específica
curl -X POST "http://localhost:8000/whale/watch" \
  -H "Content-Type: application/json" \
  -d '{"address": "0x..."}'

# Ver alertas
curl "http://localhost:8000/whale/alerts?limit=50"
```

### Tipos de Alertas

| Tipo | Severidade | Descrição |
|------|------------|-----------|
| 🐋 WHALE_DETECTED | MEDIUM-HIGH | Grande volume detectado |
| 🆕 NEW_WALLET_ACTIVITY | HIGH-CRITICAL | Carteira nova com trade grande |
| ⏰ SUSPICIOUS_TIMING | HIGH-CRITICAL | Trade perto da resolução |
| 🎯 POTENTIAL_INSIDER | CRITICAL | Múltiplos indicadores suspeitos |

### Caso de Uso: Detectar Insiders

Como no caso da captura de Maduro, onde carteiras fizeram grandes apostas horas antes do anúncio, o Whale Tracker pode:

1. **Identificar trades grandes** em mercados específicos
2. **Verificar idade da carteira** - carteiras novas são mais suspeitas
3. **Analisar timing** - trades feitos pouco antes da resolução
4. **Detectar coordenação** - múltiplas carteiras agindo juntas
5. **Gerar alertas** em tempo real

---

## 📝 Próximos Passos

1. **Teste Local**: Execute `python analyze_polymarket.py` para ver as oportunidades
2. **API**: Inicie o servidor e acesse `/docs` para explorar os endpoints
3. **Integração**: Use o módulo Python em seus próprios scripts
4. **Deploy**: Configure no Railway para acesso remoto
5. **Whale Tracker**: Acesse `http://localhost:8080/whale-tracker.html` para monitorar insiders

---

**Dúvidas?** Consulte:
- `ANALISE_ARBITRAGEM.md` - Documentação do analisador
- `README.md` - Visão geral do projeto
- `/docs` - Documentação da API (quando servidor estiver rodando)



