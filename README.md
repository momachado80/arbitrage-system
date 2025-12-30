# Kalshi-Polymarket Arbitrage System

Sistema de detecção e avaliação de arbitragem entre mercados de previsão Kalshi e Polymarket.

## Filosofia do Projeto

**Este não é um sistema que busca diferenças de preço.**

É um motor de decisão econômica sob incerteza que responde continuamente:

> "Estes dois mercados são economicamente equivalentes o suficiente para permitir arbitragem sem risco de base inaceitável?"

### Princípios Fundamentais

1. **Se houver dúvida, rejeite**
2. **Se algo parecer bom demais, rejeite**
3. **Se não conseguir explicar claramente o edge, rejeite**
4. **Um sistema que encontra muitas arbitragens rapidamente está ERRADO**

## Arquitetura

```
src/
├── models/          # Entidades de dados (Market, OrderBook, Side, etc.)
├── collectors/      # Coleta de dados (Kalshi, Polymarket)
├── engine/          # Motor de cálculo (ExecutionSimulator, ArbitrageCalculator)
└── main.py          # Integração e CLI
```

### Camadas do Sistema

1. **Coleta de Dados** - Leitura dos livros de ordens
2. **Simulação de Execução** - Cálculo de VWAP real (nunca top of book)
3. **Cálculo de Arbitragem** - Fees, slippage, penalidades
4. **Decisão** - Aceitar ou rejeitar com razão explícita

## Instalação

```bash
cd arbitrage-system
pip install -r requirements.txt
```

## Uso

### Demo de Coleta de Dados

```bash
python -m src.main demo
```

### Como Módulo Python

```python
import asyncio
from decimal import Decimal
from src.main import ArbitrageScanner

async def run():
    # Criar scanner com target de $20
    scanner = ArbitrageScanner(target_usd=Decimal("20"))
    
    # Adicionar pares de mercados MANUALMENTE VERIFICADOS
    # Equivalência contratual deve ser confirmada antes de adicionar
    scanner.add_market_pair(
        kalshi_ticker="TICKER-KALSHI",
        polymarket_condition_id="0x...",
        equivalence_score=Decimal("1.0")  # 1.0 = equivalente confirmado
    )
    
    # Executar scan
    opportunities = await scanner.scan()
    
    # Provavelmente vai retornar lista vazia - isso é ESPERADO
    print(f"Oportunidades encontradas: {len(opportunities)}")

asyncio.run(run())
```

## Modelo de Dados

### Market
Representa um contrato específico em uma plataforma:
- Identificação (ticker, platform)
- Regras de resolução
- Fonte oficial de verdade
- Orderbooks (YES e NO)

### OrderBook
Livro de ordens normalizado:
- Bids (ofertas de compra) - ordenados do maior para menor
- Asks (ofertas de venda) - ordenados do menor para maior
- Best bid/ask, spread, midpoint

### Side (YES/NO)
**Não são strings.** São instrumentos financeiros:
- YES price X implica NO price (1-X)
- Podem ser replicados sinteticamente

### ExecutionSimulation
Resultado de simulação de execução:
- VWAP (Volume Weighted Average Price)
- Número de contratos preenchidos
- Níveis consumidos
- Se execução é completa

## Cálculo de Arbitragem

Para uma oportunidade ser válida, TODAS estas condições devem ser verdadeiras:

1. ✅ Contratos representam o MESMO evento econômico
2. ✅ Regras de resolução compatíveis
3. ✅ Execução simulada nos livros reais é lucrativa
4. ✅ Lucro excede limiar mínimo ajustado ao risco
5. ✅ Liquidez permite execução no tamanho desejado
6. ✅ Sem bloqueios práticos de capital

### Fórmula de Lucro Líquido

```
Lucro Líquido = Payout Garantido 
              - Custo Total (VWAP × Contratos em ambas plataformas)
              - Taxas Kalshi (0.07 × C × P × (1-P))
              - Taxas Polymarket (~0.01%)
              - Penalidade de Risco de Base
              - Buffer de Slippage
```

Se `Lucro Líquido <= 0`, a oportunidade é **REJEITADA**.

## Taxas

### Kalshi
- Taker: `0.07 × contratos × preço × (1-preço)`
- Maker: `0.0175 × contratos × preço × (1-preço)`

### Polymarket
- ~0.01% do valor (muito baixo)

## Configuração

Parâmetros ajustáveis em `ArbitrageCalculator`:

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `slippage_buffer` | 0.5% | Buffer adicional para slippage |
| `basis_risk_penalty` | 1% | Penalidade por risco de base |
| `min_edge_threshold` | 2% | Edge mínimo para aceitar |

## Próximos Passos (Roadmap)

1. ✅ Coleta e normalização de livros
2. ✅ Simulador de execução robusto
3. ⬜ Camada manual de equivalência contratual
4. ⬜ Interface para adicionar pares
5. ⬜ Modo paper trading com logs
6. ⬜ Calibração de parâmetros
7. ⬜ Automação gradual

## Critério de Sucesso

O MVP é bem-sucedido se:

- ✅ Rejeita a maioria das oportunidades aparentes
- ✅ Explica claramente cada rejeição
- ⬜ As poucas aceitas sobrevivem à simulação
- ⬜ Diferença entre previsto e executado é pequena

## Aviso Legal

Este sistema é para fins educacionais e de pesquisa. Trading em mercados de previsão envolve riscos significativos. Não é aconselhamento financeiro.
# trigger redeploy
