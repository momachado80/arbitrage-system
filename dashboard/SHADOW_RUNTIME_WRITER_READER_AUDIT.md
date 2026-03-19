# Shadow Runtime Writer/Reader Audit

## 1. Causa raiz

A inconsistência — `closedTradeCount` e `realizedPnlTotal` subindo enquanto `lastCycleProcessedAt` permanece null e funnel zerado — tem duas causas prováveis:

### A. Separação writer/reader (multi-instância)

- **Writer**: processo/instância que executa `runCycle`, chama `closeShadowTrade` e persiste em arquivo.
- **Reader**: processo/instância que serve `/api/shadow/audit`, lê `profileStates` em memória.

Se o writer e o reader forem instâncias diferentes (ex.: Railway com réplicas ou serviço worker separado):
- O writer incrementa `closedTrades` e persiste.
- O reader carrega dados do arquivo na reidratação (apenas no boot).
- O reader nunca executa `closeShadowTrade`, logo `lastCycleProcessedAt` e funnel permanecem zerados.
- Aumento de closes no reader ocorre apenas quando **restart** carrega um arquivo já atualizado pelo writer.

### B. Close path sem heartbeat/funnel

- `closeShadowTrade` era o único ponto que alterava `closedTrades` e `realizedPnL`.
- `updateProfileHeartbeat` e `record*` do funnel eram chamados só em `runCycle`, após o processamento de cada profile.
- Em tese, todo close acontece em `runCycle`, então heartbeat e funnel deveriam ser atualizados.
- Na prática, se writer e reader forem processos distintos, o reader não vê essas mutações.

## 2. Por que a subida de closes no fillrelax derrota a explicação puramente de reidratação

- Reidratação roda **uma vez** no boot e carrega o estado do arquivo naquele momento.
- Se `closedTradeCount` do fillrelax passou de 4 para 15 **sem restart**, isso implica:
  1. Ou o **mesmo processo** está fechando trades (e nesse caso heartbeat e funnel também seriam atualizados), ou
  2. Requisições diferentes foram atendidas por **instâncias diferentes** (uma com 4 e outra com 15, por exemplo).

- Com múltiplas instâncias:
  - Instância A pode ter reidratado com 4 closes.
  - Instância B pode ter reidratado depois, com 15 closes.
  - O usuário vê “4 → 15” ao alternar entre instâncias, não por mutação em um único processo.

## 3. Impacto na comparação pai vs challenger

- Sem consistência entre writer e reader, não há garantia de que os dados do audit correspondam ao processo que está fechando trades.
- `profileObservabilityConsistency` e `shadowRuntimeConsistencyDebug` indicam se o processo que respondeu ao audit é o writer ou um reader.

## 4. Correção aplicada

1. **`closeShadowTrade` passa a atualizar heartbeat**:
   - Atualiza `lastCycleProcessedAt` e chama `recordHeartbeatMutation(profileId)`.
   - Garante que, no processo que fecha trades, todo close gera heartbeat.

2. **Bloco `shadowRuntimeConsistencyDebug`** em `/api/shadow/audit`:
   - `runtimeInstanceId`, `processUptimeMs`, `auditServedAt`
   - `lastEconomicMutationAt`, `lastHeartbeatMutationAt`, `lastFunnelMutationAt`
   - `economicMutationsCount`, `heartbeatMutationsCount`, `funnelMutationsCount`
   - `writerReaderConsistencyStatus`: `CONSISTENT`, `ECONOMICS_WITHOUT_HEARTBEAT`, `ECONOMICS_WITHOUT_FUNNEL`, `READER_ONLY`, `INCONCLUSIVE`
   - `byProfile`: `lastCloseMutationAt`, `lastHeartbeatAt`, `lastFunnelCounterMutationAt`, flags de mudança desde boot

3. **Instrumentação de mutações**:
   - `recordEconomicMutation` em `closeShadowTrade`
   - `recordHeartbeatMutation` em `updateProfileHeartbeat` e em `closeShadowTrade`
   - `recordFunnelMutation` em `markCounterMutation` (profileEligibilityFunnel)

4. **Diagnóstico de multi-instância**:
   - `READER_ONLY`: economia em memória mas nenhum close neste processo.
   - `ECONOMICS_WITHOUT_HEARTBEAT`: closes neste processo sem heartbeat (não deve ocorrer após o fix).
   - `CONSISTENT`: economia, heartbeat e funnel mutando no mesmo processo.

## 5. Como validar em produção

```bash
# shadowRuntimeConsistencyDebug
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.shadowRuntimeConsistencyDebug'

# Status de consistência writer/reader
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '{
  writerReaderConsistencyStatus: .shadowRuntimeConsistencyDebug.writerReaderConsistencyStatus,
  economicMutationsCount: .shadowRuntimeConsistencyDebug.economicMutationsCount,
  heartbeatMutationsCount: .shadowRuntimeConsistencyDebug.heartbeatMutationsCount,
  funnelMutationsCount: .shadowRuntimeConsistencyDebug.funnelMutationsCount,
  summary: .shadowRuntimeConsistencyDebug.summary
}'

# Por profile
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.shadowRuntimeConsistencyDebug.byProfile["shadow_1000_structural_fillrelax_v1"]'

# Verificar se IDs de instância mudam entre requests (multi-instância)
for i in 1 2 3; do
  curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq -r '.shadowRuntimeConsistencyDebug.runtimeInstanceId'
  sleep 1
done
```
