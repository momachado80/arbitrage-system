# Eligibility Counter Wiring Audit

## 1. Causa raiz

A inconsistência observada — `hasHeartbeat=true`, `hasEconomicHistory=true`, `cyclesProcessed=0`, `rawOpportunitiesSeen=0` — tem **causa raiz única**:

**Reidratação setava `lastCycleProcessedAt` incorretamente.**

Em `shadowSimulationStore.ts`, a função `rehydrateFromPersistence()` restaura trades fechados do disco. Para cada profile com trades restaurados, o código fazia:

```ts
state.lastCycleProcessedAt = new Date().toISOString();
```

Isso gerava um **heartbeat falso**: o profile parecia "processado neste runtime" sem nunca ter passado por `runCycle`. O funnel (`profileEligibilityFunnel.ts`) vive em memória, não é persistido e nunca é tocado pela reidratação. Resultado:

- `lastCycleProcessedAt` preenchido (da reidratação) → `hasHeartbeat = true`
- `closedTrades` e `realizedPnL` reidratados → `hasEconomicHistory = true`
- `funnelState` em memória, nunca atualizado neste processo → `cyclesProcessed = 0`, `rawOpportunitiesSeen = 0`

## 2. Por que 1 hora de espera derrota a hipótese de timing

Com ciclo de ~10s, 1 hora implica ~360 ciclos. Se o funil fosse preenchido pelo `runCycle`, os counters já teriam sido incrementados. O fato de permanecerem zerados após tanto tempo mostra que:

1. O heartbeat **não** vinha de `runCycle` — vinha da reidratação.
2. Ou o `runCycle` não estava rodando neste processo (por exemplo, multi-instância), e o audit lia de outra instância.

A correção (remover o heartbeat na reidratação) elimina o falso positivo e torna a leitura coerente.

## 3. Impacto na leitura pai vs challenger

Antes da correção:

- Pai e challenger apareciam com `hasHeartbeat=true` (falso) e funil zerado.
- A comparação econômica ficava confusa: PnL reidratado vs funil vazio.

Depois da correção:

- Se `runCycle` nunca rodou neste runtime: `hasHeartbeat=false`, funil zerado, economia reidratada. Status: `FUNNEL_MISSING_BUT_ECONOMICS_PRESENT` — legível.
- Se `runCycle` rodou: `hasHeartbeat=true`, funil preenchido. Status: `CONSISTENT` — comparação válida.

## 4. Correção aplicada

1. **Remoção do heartbeat falso na reidratação** (`shadowSimulationStore.ts`): não setar mais `lastCycleProcessedAt` ao restaurar trades. O heartbeat passa a ser definido apenas em `updateProfileHeartbeat()`, chamado por `runCycle` depois do processamento de cada profile.

2. **Bloco `eligibilityCounterRuntimeDebug`** em `/api/shadow/audit`:
   - `processIdentity`, `runtimeInstanceId`
   - `storeInitializedAt`, `lastAnyCounterMutationAt`
   - `totalCyclesRecorded`, `totalRawOpportunitiesRecorded`
   - `profilesWithNonZeroCycles`, `profilesWithNonZeroRawOpportunities`
   - `shadowLoopHeartbeatCount`
   - `byProfile`: `lastCounterMutationAt`, `countersEverUpdated`, `cyclesProcessed`, `rawOpportunitiesSeen`
   - `summary`: diagnóstico de consistência entre loop e funil

3. **Tracking de mutations** em `profileEligibilityFunnel.ts`: cada `record*` marca `lastCounterMutationAt` e `countersEverUpdated` para facilitar debug.

## 5. Como validar em produção

```bash
# Bloco eligibilityCounterRuntimeDebug
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.eligibilityCounterRuntimeDebug'

# Resumo do diagnóstico
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.eligibilityCounterRuntimeDebug.summary'

# Consistência loop vs funnel
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '{
  shadowLoopHeartbeatCount: .eligibilityCounterRuntimeDebug.shadowLoopHeartbeatCount,
  totalCyclesRecorded: .eligibilityCounterRuntimeDebug.totalCyclesRecorded,
  totalRawOpportunitiesRecorded: .eligibilityCounterRuntimeDebug.totalRawOpportunitiesRecorded,
  summary: .eligibilityCounterRuntimeDebug.summary
}'

# Heartbeat por profile (deve ser null até runCycle rodar)
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/profiles" | jq '.profiles[] | select(.profileId | test("structural")) | {profileId, lastCycleProcessedAt}'
```

## Diagnóstico por profile

O bloco `eligibilityCounterRuntimeDebug.byProfile` permite distinguir:

1. **Heartbeat atualiza, counters nunca mutam**: `lastCounterMutationAt` null, `countersEverUpdated` false, mas `lastCycleProcessedAt` no profile preenchido → indica que o heartbeat vem de outra fonte (ex.: reidratação antiga) ou de outro processo.

2. **Counters mutam, mas endpoint não enxerga**: `shadowLoopHeartbeatCount > 0` e `totalCyclesRecorded > 0` na resposta indicam que a instância que serviu o request tem dados; se em outra chamada vier zerado, pode ser multi-instância.

3. **Counters zeram repetidamente**: não deveria acontecer; `funnelState` não é resetado. Se ocorrer, suspeitar de múltiplas instâncias ou reinícios frequentes.

4. **Counters só existem localmente**: `runtimeInstanceId` e `processIdentity` ajudam a ver se requests diferentes atingem instâncias diferentes (IDs diferentes em respostas diferentes).
