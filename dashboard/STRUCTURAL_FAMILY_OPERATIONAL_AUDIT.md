# Auditoria operacional: família estrutural

**Data:** 2026-03  
**Objetivo:** Descobrir por que perfis estruturais pararam de atualizar (~21:02:27Z) enquanto shadow_1000 e adapt continuaram.

---

## A. Causa mais provável

**Causa:** `lastUpdate` só era atualizado em **open** ou **close**. Perfis estruturais, com filtros rígidos (pair set, fill bucket, capfloor, degratio), deixaram de ter oportunidades compatíveis e de ter trades ativos. Sem open/close, `lastUpdate` permaneceu no valor da reidratação (deploy ~21:02), simulando congelamento.

**Não é bug de execução:** O ciclo processa todos os perfis. O problema era semântico: `lastUpdate` não refletia “último ciclo processado”, só “último open/close”.

---

## B. Evidências

### 1. Código

- **shadowSimulationStore.ts:** `lastUpdate` era definido em `addShadowTrade`, `closeShadowTrade` e reidratação.
- **updateShadowUnrealized:** não altera `lastUpdate`.
- **Conclusão:** Se um perfil não abre nem fecha trades, `lastUpdate` não muda.

### 2. Diferença estrutural vs não estrutural

| Aspecto | Estruturais | Não estruturais |
|---------|-------------|------------------|
| Filtros | pair set, fill 0.1–0.25, capfloor 4.5%, degratio 0.24 | Sem restrições estruturais |
| Ocorrência de opps compatíveis | Baixa | Alta |
| Open/close | Raro | Frequente |
| Atualização de `lastUpdate` | Rara | Frequente |

### 3. Momento do congelamento

- Se houve deploy/reinício ~21:02, a reidratação definiu `lastUpdate` para esse momento em todos os perfis com trades fechados.
- Perfis estruturais: sem novos opens e com `activeTrades = 0` após fechamentos.
- `lastUpdate` permaneceu em 21:02.
- Perfis não estruturais: continuaram abrindo/fechando → `lastUpdate` seguiu atualizando.

---

## C. Arquivos analisados

- `lib/shadowSimulationService.ts` (fluxo de ciclo, try/catch por perfil)
- `lib/shadowSimulationStore.ts` (onde e quando `lastUpdate` é atualizado)
- `lib/shadowSimulationProfiles.ts` (configuração dos perfis estruturais)
- `lib/structuralChallengerHelpers.ts`, `structuralRiskManagedDiagnostics.ts`

---

## D. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `lib/shadowSimulationStore.ts` | Campo `lastCycleProcessedAt`, `updateProfileHeartbeat()` |
| `lib/shadowSimulationService.ts` | Chamada `updateProfileHeartbeat()` ao fim do processamento de cada perfil |
| `lib/structuralFamilyOperationalDiagnostics.ts` | **NOVO** – diagnósticos operacionais por perfil/família |
| `app/api/shadow/audit/route.ts` | Exposição de `structuralFamilyOperationalDiagnostics` |
| `app/api/shadow/profiles/route.ts` | Exposição de `lastCycleProcessedAt` |

---

## E. Commit e validação em produção

```bash
# Diagnósticos operacionais
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralFamilyOperationalDiagnostics'

# Heartbeat por perfil
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/profiles" | jq '.profiles[] | {profileId, lastUpdate, lastCycleProcessedAt}'
```

---

## F. Recomendação

- **Problema operacional:** não é falha de execução; o ciclo processa todos os perfis.
- **Ausência de oportunidades:** perfis estruturais tendem a ter menos opps compatíveis. O “congelamento” era `lastUpdate` parado por falta de open/close.
- **Com a correção:** `lastCycleProcessedAt` passa a ser atualizado em todo ciclo. Se ainda houver perfis estruturais com heartbeat stale enquanto não estruturais estiverem vivos, investigar exceções no try/catch (logs `[ShadowSim] profile X failed`).
