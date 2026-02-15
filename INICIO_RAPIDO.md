# 🚀 Início Rápido - Sistema de Arbitragem

## Forma Mais Rápida de Acessar

### 1️⃣ Análise de Arbitragem Polymarket (Recomendado)

```bash
python analyze_polymarket.py
```

**Isso vai:**
- ✅ Conectar ao Polymarket
- ✅ Escanear mercados em tempo real
- ✅ Filtrar por volume > $100k
- ✅ Identificar divergências de probabilidade
- ✅ Mostrar Top 3 oportunidades com maior EV

---

### 2️⃣ API REST (Para Integração)

```bash
# Iniciar servidor
uvicorn api:app --reload

# Em outro terminal, testar:
curl http://localhost:8000/polymarket/analyze

# Ou abrir no navegador:
# http://localhost:8000/docs
```

---

### 3️⃣ Interface Web

```bash
# Iniciar API primeiro
uvicorn api:app --reload

# Abrir frontend/index.html no navegador
```

---

## 📋 Checklist de Primeira Execução

- [ ] Instalar dependências: `pip install -r requirements.txt`
- [ ] Executar: `python analyze_polymarket.py`
- [ ] Ver resultados no terminal

---

## 🎯 O Que Você Vai Ver

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TOP 3 OPORTUNIDADES DE ARBITRAGEM                        ║
║                    Polymarket - Análise em Tempo Real                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OPORTUNIDADE #1 - POLYMARKET ARBITRAGE                                      ║
║  Mercado: [Título do mercado]                                                ║
║  Preço: $0.65 | Probabilidade Esperada: 75% | Edge: 10%                    ║
║  Retornos: $10→$1.00 | $20→$2.00 | $100→$10.00 | $1000→$100.00              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📚 Documentação Completa

- **Como Acessar**: `COMO_ACESSAR.md`
- **Análise de Arbitragem**: `ANALISE_ARBITRAGEM.md`
- **API Endpoints**: http://localhost:8000/docs (quando servidor estiver rodando)

---

**Pronto para começar?** Execute: `python analyze_polymarket.py`




