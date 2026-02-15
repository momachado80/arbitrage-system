# Analisador de Arbitragem - Polymarket

## Visão Geral

Este sistema analisa mercados do Polymarket em tempo real para identificar oportunidades de arbitragem baseadas em divergências de probabilidade. O analisador atua como um **Especialista em Previsão de Mercados** e identifica situações onde o preço atual diverge significativamente da probabilidade esperada do evento.

## Critérios de Filtragem

O sistema filtra mercados com as seguintes características:

1. **Volume e Liquidez**: Apenas mercados com volume superior a **$100.000** nas últimas 24 horas
2. **Timeframe**: Prioriza mercados que resolvem em menos de **48 horas** ou mercados diários
3. **Divergência de Probabilidade**: Identifica eventos onde o preço no Polymarket está significativamente diferente da probabilidade esperada (mínimo 10% de divergência)
4. **Expected Value (EV)**: Apresenta apenas oportunidades com EV positivo (mínimo $0.05 por $1 investido)

## Como Usar

### Execução Básica

```bash
python analyze_polymarket.py
```

O script irá:
1. Conectar à API do Polymarket
2. Escanear até 200 mercados ativos
3. Filtrar por volume, timeframe e divergência
4. Calcular Expected Value para cada oportunidade
5. Apresentar os **Top 3** mercados com maior EV positivo

### Saída do Sistema

O sistema apresenta para cada oportunidade:

- **Análise de Probabilidade**:
  - Preço atual no Polymarket
  - Probabilidade esperada (estimada)
  - Divergência entre preço e expectativa
  - Edge percentual

- **Dados de Mercado**:
  - Volume 24h
  - Liquidez disponível
  - Tempo até resolução
  - Score de confiança

- **Estratégia Recomendada**:
  - Lado para apostar (YES ou NO)
  - Investimento sugerido
  - Expected Value (EV)

- **Tabela de Retornos Projetados**:
  - Retornos para investimentos de $10, $20, $100 e $1.000
  - ROI percentual
  - Tempo estimado até resolução
  - Score de confiança

## Exemplo de Saída

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  OPORTUNIDADE #1 - POLYMARKET ARBITRAGE                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Mercado: Will Bitcoin reach $100,000 by end of 2024?                        ║
║ Condition ID: 0x1234...                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ANÁLISE DE PROBABILIDADE:                                                    ║
║   Preço Polymarket:        $0.6500 (65.00%)                                 ║
║   Probabilidade Esperada:  75.00%                                            ║
║   Divergência:             10.00%                                            ║
║   Edge:                    10.00%                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ESTRATÉGIA RECOMENDADA:                                                       ║
║   Apostar em:              YES                                               ║
║   Investimento Sugerido:   $500.00                                           ║
║   Expected Value (EV):     $0.1000 por $1 investido                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ RETORNOS PROJETADOS:                                                         ║
║   ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐ ║
║   │ Investimento│   Retorno    │   ROI (%)    │   Tempo Est. │   Confiança  │ ║
║   ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤ ║
║   │    $10      │    $1.00     │   10.00%     │   24.0 horas │   75.0%      │ ║
║   │    $20      │    $2.00     │   10.00%     │   24.0 horas │   75.0%      │ ║
║   │   $100      │   $10.00     │   10.00%     │   24.0 horas │   75.0%      │ ║
║   │  $1,000     │  $100.00     │   10.00%     │   24.0 horas │   75.0%      │ ║
║   └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Como Funciona

### 1. Coleta de Dados

O sistema usa a API pública do Polymarket:
- **Gamma API**: Metadados de mercados, volume, datas de resolução
- **CLOB API**: Orderbooks em tempo real, preços, liquidez

### 2. Estimativa de Probabilidade Esperada

**IMPORTANTE**: A estimativa de probabilidade esperada usa uma heurística simplificada baseada em:
- Volume do mercado (mercados com mais volume tendem a ser mais eficientes)
- Reversão à média (ajuste em direção à probabilidade 50%)
- Padrões de mercado similares

**Para uso em produção**, você deveria integrar com:
- Odds de casas de apostas tradicionais (Betfair, Pinnacle, etc.)
- Modelos de previsão especializados
- Agregadores de odds confiáveis
- Análise de especialistas

### 3. Cálculo de Expected Value (EV)

Para cada mercado, o sistema calcula:

```
EV = (Probabilidade Esperada × Payout) - Custo Atual - Taxas

Onde:
- Payout = $1 por contrato se ganhar, $0 se perder
- Custo Atual = Preço de compra no Polymarket
- Taxas = Taxa da Polymarket (~0.01% por contrato)
```

### 4. Filtragem e Ranking

Mercados são filtrados e ordenados por:
1. Volume mínimo ($100k)
2. Timeframe máximo (48 horas)
3. Divergência mínima (10%)
4. EV mínimo ($0.05 por $1)
5. Ordenação final por EV (maior primeiro)

## Avisos Importantes

⚠️ **AVISOS LEGAIS E DE RISCO**:

1. **Este é um sistema de análise automatizada**. Sempre faça sua própria pesquisa antes de tomar decisões de investimento.

2. **As probabilidades esperadas são estimadas** usando heurísticas. Em produção, você deveria integrar com fontes externas confiáveis.

3. **Arbitragem de mercados de previsão envolve riscos**:
   - Risco de base (mercados podem não ser perfeitamente equivalentes)
   - Risco de liquidez (ordens podem não executar ao preço esperado)
   - Risco de resolução (disputas sobre resultados)
   - Risco de taxas e custos de transação

4. **Sempre teste estratégias em modo paper trading** antes de usar capital real.

5. **Este sistema é fornecido "como está"**, sem garantias de lucro ou precisão.

## Personalização

Você pode ajustar os critérios editando `src/engine/arbitrage_analyst.py`:

```python
MIN_VOLUME_USD = Decimal("100000")  # Volume mínimo
MAX_TIMEFRAME_HOURS = 48  # Timeframe máximo
MIN_DIVERGENCE = Decimal("0.10")  # Divergência mínima (10%)
MIN_EV = Decimal("0.05")  # EV mínimo por $1 investido
```

## Dependências

- Python 3.8+
- httpx (cliente HTTP assíncrono)
- Ver `requirements.txt` para lista completa

## Licença

Este sistema é fornecido para fins educacionais e de pesquisa. Use por sua conta e risco.




