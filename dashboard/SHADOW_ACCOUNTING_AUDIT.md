# Auditoria Contábil — Shadow Portfolio

**Data:** 2025-02-05  
**Objetivo:** Rastrear fórmulas, ciclo de vida e caminhos que reduzem equity; identificar causa do colapso para ~1e-13.

---

## 1. FÓRMULAS EXATAS DE CONTABILIDADE

### 1.1 Variáveis de estado (`shadowSimulationStore.ts`)

| Variável | Fórmula / Definição | Local |
|----------|---------------------|-------|
| `startingCapital` | Config fixa. `config.startingCapital` (500 ou 5000) | `initProfileState` L84 |
| `currentEquity` | `startingCapital + realizedPnL + unrealizedPnL` | L185, L209 |
| `realizedPnL` | Acumulado: `+= closed.realizedPnL` em cada close | L177 |
| `unrealizedPnL` | Substituído: `unrealizedPnL = unrealized` (soma mtm dos ativos) | L208 |
| `reservedCapital` | `Σ trade.filledCapital` para trades em `activeTrades` | L141 (+), L178 (-) |
| `availableCapital` | `max(0, currentEquity - reservedCapital)` | L148, L186, L210 |

### 1.2 Invariantes

- `currentEquity` **não** usa `max(0, ...)` — pode ficar negativo em teoria.
- `availableCapital` sempre ≥ 0.
- `reservedCapital` é mantido incrementalmente (add/close), não recalculado a partir de `activeTrades`.

---

## 2. CICLO DE VIDA DE UM SHADOW TRADE

### 2.1 OPEN (abertura)

**Onde:** `shadowSimulationService.ts` (runCycle L207–230, evaluateOpportunity L456–474)

1. `simulateRealisticEntry` calcula `filledCapital`, `effectiveEntryPrice`.
2. Trade criado com `realizedPnL: 0`, `status: "active"`.
3. `addShadowTrade(profileId, trade, config)`:
   - `activeTrades.push(trade)` (L140)
   - `reservedCapital += trade.filledCapital` (L141)
   - `availableCapital = max(0, currentEquity - reservedCapital)` (L148)
   - **Não altera** `currentEquity`, `realizedPnL` nem `unrealizedPnL`.

### 2.2 RESERVA DE CAPITAL

- Reserva: `reservedCapital += filledCapital` em `addShadowTrade`.
- Capital disponível diminui via `availableCapital = currentEquity - reservedCapital`.
- `currentEquity` não muda na abertura.

### 2.3 MARK TO MARKET (unrealized)

**Onde:** `shadowSimulationService.ts` L287–292

```ts
unrealized = Σ (para cada t em activeTrades) {
  exitEst = 1 - t.observedEdgeAtEntry;
  pnl = t.filledCapital * ((exitEst - t.effectiveEntryPrice) / max(0.001, t.effectiveEntryPrice));
  return s + pnl;
}
updateShadowUnrealized(profileId, unrealized);
```

- `exitEst` usa o edge **na entrada**, não preço de mercado atual.
- `updateShadowUnrealized` substitui `unrealizedPnL` e recalcula `currentEquity`.

### 2.4 CLOSE (fechamento)

**Onde:** `shadowSimulationService.ts` L267–281, `closeShadowTrade` L155–202

1. `simulateRealisticExit` calcula `realizedPnL`, `effectiveExitPrice`.
2. `closeShadowTrade(profileId, tradeId, { realizedPnL, ... })`:
   - Remove trade de `activeTrades` (splice)
   - Adiciona em `closedTrades`
   - `realizedPnL += closed.realizedPnL ?? 0` (L177)
   - `reservedCapital -= t.filledCapital` (L178)
   - `currentEquity = startingCapital + realizedPnL + unrealizedPnL` (L185)
   - `availableCapital = max(0, currentEquity - reservedCapital)` (L186)

### 2.5 RELEASE DE CAPITAL

- Liberação: `reservedCapital -= filledCapital` em `closeShadowTrade` (L178).
- Aplicação de PnL: `realizedPnL += closed.realizedPnL` (L177).

### 2.6 ARQUIVAMENTO / CLEANUP

- `closedTrades.slice(-500)` (L174–175): mantém só os últimos 500.
- Não altera `realizedPnL` (soma acumulada permanece).
- `equityCurve.slice(-200)` para histórico (L196–197).

---

## 3. FÓRMULA DE REALIZED PnL POR TRADE

**Onde:** `realisticExecutionEngine.ts` L347–352

```ts
exitPrice = latestOpportunity ? 1 - latestOpportunity.edge : activeTrade.effectiveEntryPrice;
effectiveExitPrice = exitPrice - exitImpactResult.expectedPriceWorseningExit;
pnlPct = (effectiveExitPrice - activeTrade.effectiveEntryPrice) / max(0.001, activeTrade.effectiveEntryPrice);
realizedPnL = activeTrade.filledCapital * pnlPct;
```

---

## 4. TODOS OS CAMINHOS QUE REDUZEM EQUITY

| Caminho | Efeito | Local |
|---------|--------|-------|
| **1. `realizedPnL` negativo** | Cada close com perda reduz `currentEquity` | `closeShadowTrade` L177, L185 |
| **2. `unrealizedPnL` negativo** | Mark-to-market de posições em prejuízo | `updateShadowUnrealized` L208–209 |
| **3. Nenhum outro** | Não há reset, nem dupla dedução direta de equity | — |

---

## 5. ANÁLISE: POR QUE EQUITY CAI DE 500/5000 PARA ~1e-13?

### 5.1 Consistência matemática

Com `currentEquity = startingCapital + realizedPnL + unrealizedPnL`:

- Se `realizedPnL ≈ -500` (500 trades fechados com perda média ~1 por unidade) e `unrealizedPnL` pequeno:
  - `currentEquity ≈ 500 + (-500) + unrealizedPnL ≈ 0 + unrealizedPnL`
- Com 10 trades ativos e mtm pequeno, `unrealizedPnL` da ordem de 1e-12 a 1e-13 é plausível.
- Valores como `3.98e-13` são compatíveis com ruído de ponto flutuante quando equity ≈ 0.

### 5.2 Dados de produção

- `shadow_100`: start=500, equity≈3.98e-13, reserved≈3.77e-13, avail≈2.09e-14, active=10, closed=500
- `shadow_1000`: start=5000, equity≈8.19e-12, reserved≈7.86e-12, avail≈3.28e-13, active=10, closed=500

Com 500 closes e equity ≈ 0, a explicação mais provável:

1. Perdas reais acumuladas em `realizedPnL` que zeraram o capital.
2. Os ~10 trades ativos têm `filledCapital` muito pequeno (ordem 1e-12 a 1e-13), pois `availableCapital` já era quase zero quando foram abertos.

### 5.3 Comportamento esperado ou bug?

- **Contabilidade:** coerente. Não há indício de:
  - capital deduzido duas vezes,
  - perdas aplicadas mais de uma vez,
  - reserva não liberada,
  - trades fechados ainda influenciando equity,
  - uso de valores malformados no cálculo de equity.

- **Resultado:** o colapso de equity parece consequência de muitas operações perdedoras, não de erro contábil.

---

## 6. COMPORTAMENTOS SUSPEITOS AVALIADOS

| Suspeita | Verificação | Resultado |
|----------|-------------|-----------|
| Capital deduzido duas vezes | Só `reservedCapital += filledCapital` no add e `-= filledCapital` no close | Ok |
| Perdas aplicadas repetidamente | `realizedPnL += closed.realizedPnL` uma vez por close | Ok |
| Reserva não liberada | `reservedCapital -= t.filledCapital` em todo close | Ok |
| Trades fechados afetando equity | Removidos de `activeTrades`; só somam em `realizedPnL` | Ok |
| Equity de valores malformados | Fórmula usa apenas `startingCapital`, `realizedPnL`, `unrealizedPnL` | Ok |
| Compounding / multiplicação ≈ 0 | `deterministicFill` e PnL usam multiplicações padrão | Ok |
| `reservedCapital` dessincronizado | Mantido incrementalmente; invariante `reserved ≈ Σ active.filledCapital` | Ok |

### 6. Risco de ponto flutuante

- Muitos `+=` e `-=` em `reservedCapital` e `realizedPnL` podem gerar erros de arredondamento.
- Valores tipo `3.98e-13` quando equity deveria ser 0 são compatíveis com esse ruído.

---

## 7. CONCLUSÕES E PRÓXIMOS PASSOS

### 7.1 Resumo

1. **Fórmulas:** definidas e consistentes.
2. **Ciclo de vida:** open → reserve → mtm → close → release → archive, bem delimitado.
3. **Redução de equity:** apenas via `realizedPnL` negativo e `unrealizedPnL` negativo.
4. **Causa mais provável:** perdas realizadas acumuladas em ~500 closes, levando equity para ≈ 0, com valor residual ~1e-13 explicado por ponto flutuante.
5. **Tipo:** comportamento esperado da lógica contábil, não bug de contabilidade.
6. **Correção mais segura (se quiser evidência extra):** adicionar diagnósticos mínimos de accounting para registrar `realizedPnL`, `unrealizedPnL` e `currentEquity` em cada close e em cada `updateShadowUnrealized`.

### 7.2 Diagnósticos sugeridos (opcional)

Para confirmar a hipótese com dados reais:

- Em cada `closeShadowTrade`: log `{ closedCount, realizedPnLDelta, stateRealizedPnL, stateCurrentEquity }`.
- Em cada `updateShadowUnrealized`: log `{ unrealized, stateCurrentEquity }`.

Sem alterar lógica de negócio, ranking, execução nem thresholds.
