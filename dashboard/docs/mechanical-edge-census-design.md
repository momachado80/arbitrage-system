# Mechanical Edge Census (MEC) — Design v1

**Status:** MEC-1 implementado (lib pura + testes). Runner/Tier 3 pendentes.
**Governança:** shadow-only, read-only, 1 hipótese por vez. Sem execução, paper, microcapital, bot global.

## 1. Objetivo e não-objetivos

**Objetivo:** detectar e medir, sobre o universo inteiro, ineficiências **mecânicas**
(incoerência de preço verificável numa única foto do livro) líquidas dos 6 custos reais,
e provar **persistência** através de snapshots — sem executar ordens.

**Não-objetivos:**
- Não prevê retorno (não é alfa estatístico — isso era a Hipótese #1, arquivada por throughput).
- Não executa, não faz paper, não toca microcapital.
- Não reusa `probabilityScanner.ts`/`opportunityEngine.ts` (mid-based + amarrados ao bot). MEC é fill-based e desacoplado.

## 2. Por que o pivô

A Hipótese #1 (post-event reversion) acumulou ~2 amostras qualificadas em ~6 semanas →
n=50 levaria ~3 anos. Morta por throughput. Edge mecânico é **auto-falsificável no dia 1**:
ou a cesta custa menos que o payout garantido líquido de custos, ou não custa.

## 3. Arquitetura — 2 tiers + persistência

```
Tier 0: snapshot universe   → mercados/eventos candidatos (Gamma)
Tier 1: cheap census        → coerência fill-based via bestAsk/bestBid reais
Tier 2: deep validation     → VWAP por profundidade + 6 custos (evaluateMecBasket)
Tier 3: persistence test    → o gap sobrevive a N snapshots ou é comido por bot?
        → Ledger JSONL → Summarizer → Verdict
```

MEC-1 (este commit) entrega **só o núcleo do Tier 2**: a função pura `evaluateMecBasket`
em `lib/mechanicalEdgeCensus.ts`, mais testes. Sem rede, sem runner, sem deploy.

## 4. Tipos de edge mecânico

| Tipo | Condição (fill-based) | Ação implícita (NÃO executada) |
|---|---|---|
| `BINARY_UNDERROUND` | `ask_yes + ask_no < 1` | comprar ambos → payout garantido 1 |
| `BINARY_OVERROUND` | `bid_yes + bid_no > 1` | vender ambos (precisa colateral) |
| `PARTITION_UNDERROUND` | `Σ ask_yes_i < 1` (mut. excl., exaustivo) | comprar a cesta |
| `NEGRISK_CONVERSION` | bundle NO→YES via mecânica negRisk | + fee de conversão |
| `RESOLUTION_CONVERGENCE` | preço ≠ {0,1} perto de `endDate` determinístico | segurar até settlement (v1.1) |

## 5. Fórmula de net-edge — os 6 custos

Para cesta de `k` pernas, tamanho-alvo `Q` USD:

```
gross (buy)  = max(0, 1 − Σ vwapAsk_i)      # executável, não mid
gross (sell) = max(0, Σ vwapBid_i − 1)

net = gross − cGas − cLockup − cUma − cConversion − cLegRisk
```

| Custo | Fórmula | Nota |
|---|---|---|
| slippage profundidade | embutido no gross (VWAP) | `cSlippageDepth` = \|Σvwap − Σbest\|, diagnóstico |
| `cGas` | `k · gasPerTxUsd / Q` | Polygon ~baixo |
| `cLockup` | `capitalAnnual · (days/365) · capitalPerUnit` | **o mais subestimado** |
| `cUma` | `umaHaircut[category]` | risco de resolução/disputa |
| `cConversion` | `conversionFeeFrac` | só NEGRISK |
| `cLegRisk` | `legRiskCoeff · meanSpread · √(k−1)` | risco de preencher parte da cesta |

**Conserto-chave vs código legado:** `negativeRiskConversionPilot` usava `0.55·slack` (fudge)
e mid. MEC usa VWAP real de asks → `gross` direto, sem fator de correção.

**Custos que não existiam em lib nenhuma:** `cLockup` (lockup de capital até resolução),
`cUma` (risco de disputa), e o Tier 3 de persistência (anti-bot).

## 6. Constantes calibradas (sessão 2026-06)

```
costOfCapitalAnnual = 0.10        # midpoint 8–12%
targetSizeUsd       = 100
gasPerTxUsd         = 0.03
legRiskCoeff        = 0.5
umaHaircutByCategory = {
  crypto_feed: 0.001,  sports: 0.003,  macro_data: 0.004,
  electoral:   0.006,  subjective: 0.020,  unknown: 0.010,
}
```

Filosofia UMA: **errar para cima**. Falso "viable" custa capital numa disputa; falso
"not_viable" só custa uma oportunidade que provavelmente nem existia.

## 7. Bandas de verdict (snapshot)

```
net < 0                              → negative_after_costs   (maioria esperada)
0 ≤ net < 0.5%                       → not_viable
net ≥ 0.5% & profundidade < alvo     → capacity_insufficient
0.5% ≤ net < 1.0% & capacidade ok    → marginal_pending_persistence
net ≥ 1.0% & capacidade ok           → viable_pending_persistence
```

A função pura dá verdict de **snapshot** — nunca afirma "viable_candidate". Só o Tier 3
(runtime, multi-snapshot) promove `*_pending_persistence` → `viable_candidate` após
confirmar `persistence_score ≥ 0.7` em ≥ N observações. Isso separa edge capturável por
operador pequeno (persiste) de miragem (some em segundos = comida por bot).

## 8. Reuso de infraestrutura existente

| Componente | Estado | Uso no MEC |
|---|---|---|
| `clobMicrostructure.ts` | pronto | livro real bestBid/Ask/depthTop3 |
| `finalNegativeRiskValidation31552.ts` | pronto | template de custo + stress (Tier 2) |
| `negativeRiskConversionPilot.ts` | pronto | padrão de paginação Gamma (Tier 1) |
| padrão de ledger JSONL + summarizer | pronto | persistência + verdict |

## 9. Fasing

| Fase | Entrega | Estado |
|---|---|---|
| MEC-0 | Arquivar H#1, desligar observer | pendente |
| **MEC-1** | **lib pura + testes** | **ENTREGUE** |
| MEC-2 | Runner Tier 1+2 read-only + ledger | pendente |
| MEC-3 | Tier 3 persistência | pendente |
| MEC-4 | Censo amplo 24–72h | pendente |
| MEC-5 | GO/NO-GO: existe viable_candidate recorrente? | pendente |

## 10. Garantias shadow-only

- Lib pura sem rede/I/O. Runner (futuro) só GET Gamma + GET CLOB book.
- Zero import de dispatcher/paper/shadow-store/bot global/graph.
- `canUseForExecution: false` literal em todo output.
- Ledger append-only em `/data`, nunca `.paper`.
- Passa `check:shadow-only` e `check:shadow-only:gamma-1823789`.
