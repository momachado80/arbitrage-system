# Narrow Challenger — Preparação (não promotido)

## Regra operacional

**Nenhum challenger será criado ou promovido automaticamente** até que:
1. O runtime/bootstrap/upstream esteja validado em produção
2. Exista fluxo novo real (shadowLoopStarted, cycleCompletedCount > 0)
3. A análise de ilhas locais de viabilidade encontre uma hipótese plausível

## Caminho técnico preparado

- `localViabilityPrep` no `/api/shadow/audit`: segmentação por pairKey, fillRatio, capturableEdge, exitReason, holdingRegime
- `getRehydratedTradeIds()`: separação explícita entre histórico reidratado e fluxo novo
- `computeLocalViabilitySegments()`: análise só em trades novos desta execução

## Forma do futuro narrow challenger candidato

O challenger narrow **não** nasce de tuning universal. Nasce de uma **hipótese local** como:

| Hipótese local | Challenger narrow candidato | Condição para ativação |
|----------------|-----------------------------|-------------------------|
| Subset de pares com menor destruição | `excludedPairKeys` reduzido; só pares com avgPnL > -X | localViabilityPrep.byPairKey mostrar ≥1 par com winRate > 0.5 e n ≥ 5 |
| Faixa de capturable edge viável | `minCapturableEdgeToTrade` elevado para faixa 2–5% | byCapturableEdgeBucket["2-5%"] ou [">5%"] com avgPnL > 0 |
| Faixa de fill ratio viável | `minFillRatioToTrade` elevado | byFillRatioBucket["0.5-0.75"] ou ["0.75-1.0"] com avgPnL > 0 |
| Combinação pairKey × fill ratio | Regra específica por par | byPairKeyAndFillRatio com celulas com winRate > 0.5 |

## Por que não promover agora

- Ambiente ainda inválido para inferência causal (runtime morto ou sem fluxo novo)
- Histórico reidratado não prova fluxo novo
- Nenhuma alteração econômica até validar upstream

## Próximo passo

Quando `operationalTruth.environmentValidForEconomicInference === true` e houver `localViabilityPrep.newTradesCount >= 10`, revisar os segmentos e, se existir ilha local plausível, documentar o spec exato do narrow challenger e submetê-lo a aprovação manual.
