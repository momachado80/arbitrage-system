# Shadow Validation Pipeline

Esteira automatizada para validação e julgamento disciplinado do sistema de shadow trading. Reduz trabalho manual de checagem operacional, deploy, runtime e leitura inicial de challenger.

---

## Scripts

| Script | O que faz |
|--------|-----------|
| `shadow:preflight` | Valida diretório, branch, git, presença do profile e blocos no código |
| `shadow:deploy-check` | Consulta produção `/api/shadow/audit` e valida se o deploy está atualizado |
| `shadow:snapshot` | Gera `reports/runtime_snapshot_latest.json` com métricas e análise de defesas |
| `shadow:brief` | Gera `reports/decision_brief_latest.md` com brief executivo e decisão padronizada |
| `shadow:judge` | Classifica challenger em status fechado (ex: WORSE_THAN_BASELINE, PROMISING_BUT_EARLY) |
| `shadow:full-check` | Executa preflight → deploy-check → snapshot → brief → judge em sequência |

---

## Como rodar

Todos os comandos devem ser executados a partir de `dashboard/`:

```bash
cd dashboard

# Validação rápida (código local)
npm run shadow:preflight

# Verificar se produção está atualizada
npm run shadow:deploy-check

# Snapshot de runtime (requer produção acessível)
npm run shadow:snapshot

# Brief executivo (requer snapshot prévio)
npm run shadow:brief

# Julgamento (requer snapshot)
npm run shadow:judge

# Esteira completa (inclui judge)
npm run shadow:full-check
```

### URL de produção

Para `deploy-check` e `snapshot`, a URL é resolvida nesta ordem:

1. `AUDIT_URL` ou `DASHBOARD_URL` (variáveis de ambiente)
2. `dashboard/.production-url` (arquivo com uma linha: URL base)
3. Candidatos conhecidos (Railway, etc.)

Exemplo:

```bash
# Via env
AUDIT_URL=https://dashboard-next-production-b126.up.railway.app npm run shadow:deploy-check

# Ou criar dashboard/.production-url
echo "https://dashboard-next-production-b126.up.railway.app" > dashboard/.production-url
```

---

## Validade da leitura econômica

O snapshot e o brief incluem `economic_read_valid` e `economic_read_reason`:

| Condição | economic_read_valid | economic_read_reason |
|----------|---------------------|----------------------|
| Deploy stale (diagnostics ausente) | false | `deploy_stale` |
| challengerClosed = 0 | false | `challenger_closed_zero` |
| closed < 10 (configurável) | false | `sample_too_small` |
| Amostra ok | true | `ok` |

**Thresholds configuráveis** (env vars):

- `SHADOW_MIN_CLOSED_READABLE` — mín. closed para leitura válida (default: 10)
- `SHADOW_MIN_CLOSED_COMPARE` — mín. closed para comparar com baseline (default: 20)
- `SHADOW_MIN_IMPROVEMENT_PCT` — mín. melhoria relativa para LESS_BAD (default: 0.1)

---

## Status do shadow:judge

O script `shadow:judge` classifica o challenger em um enum fechado. O judgment inclui `evidenceGrade` (NONE | WEAK | MODERATE | STRONG) e `dominantFailureMode`.

### Tabela completa: status, significado, requisitos mínimos, riscos

| Status | Significado | Requisitos mínimos | Riscos de interpretação indevida |
|--------|-------------|--------------------|-----------------------------------|
| `NOT_READABLE_YET` | Dados insuficientes para qualquer conclusão | deploy_stale OU challengerClosed = 0 | Confundir com "challenger ruim" — é falta de dados |
| `SAMPLE_TOO_SMALL` | Amostra existe mas abaixo do mínimo para leitura econômica | closed > 0 E closed < 10 | Ler sinal econômico — amostra não permite |
| `OPERABLE_BUT_UNPROVEN` | Leitura econômica válida, mas sem superioridade clara vs baseline | economic_read_valid true; comparação insuficiente ou banda de ruído | Concluir "não funciona" — pode ser ruído |
| `WORSE_THAN_BASELINE` | Challenger pior que baseline por threshold | closed ≥ 20; melhoria relativa ≤ -10% | Generalizar — regime pode ter mudado |
| `LESS_BAD_THAN_BASELINE` | Challenger pelo menos 10% menos ruim que baseline | closed ≥ 20; melhoria relativa ≥ 10% | Superpromover — amostra pode ser sortuda |
| `DEFENSIVE_LOGIC_NOT_ENGAGED` | Defesas adaptativas não acionaram; challenger ruim ou inconclusivo | avgMult ≥ 0.95, early=0, capfloor=0, degratio=0; challenger negativo ou sample pequeno | Ignorar — indica que o código de defesa não está tendo efeito |
| `PROMISING_BUT_EARLY` | Direção positiva com critérios rígidos | closed < 20; challenger melhor que baseline por threshold; ≥1 defesa ativada | Tratar como evidência forte — ainda é early |

### evidenceGrade

| Grade | Condição |
|-------|----------|
| NONE | deploy stale, ou closed = 0, ou closed < 10 |
| WEAK | 10 ≤ closed < 20 |
| MODERATE | closed ≥ 20 |
| STRONG | closed ≥ 50 |

### dominantFailureMode

| Modo | Quando |
|------|--------|
| `deploy_not_live` | structuralRiskManagedDiagnostics ausente |
| `no_closed_sample` | challengerClosed = 0 |
| `structural_gate_too_narrow` | 90%+ das opps rejeitadas por pair; amostra pequena |
| `defenses_not_engaged` | multiplier ≥ 0.95, early=0, capfloor=0, degratio=0; challenger ruim |
| `worse_than_baseline` | Challenger pior que baseline por threshold |
| `insufficient_comparison_sample` | closed < 20 ou sample abaixo do ideal |
| `no_clear_failure_mode` | Evidência ok, sem causa dominante identificada |

A classificação é conservadora: nunca superpromove com pouca amostra.

---

## Análise de ativação das defesas

O snapshot e o brief incluem `defenseActivation`:

| Defesa | Indicador de ativação |
|--------|------------------------|
| Gate estrutural (pair) | `rejectedByStructuralPairCount > 0` |
| Capfloor | `rejectedByCapfloorCount > 0` |
| Degratio | `rejectedByDegRatioCount > 0` |
| Capital multiplier | `avgCapitalMultiplierOpened < 1.0` |
| Early thesis failure | `earlyThesisFailureExitCount > 0` |

Objetivo: distinguir defesas que realmente morderam das que só existem no código.

---

## Uso do shadow:judge com shadow:full-check

O `shadow:full-check` já inclui o judge como último passo. Sequência:

1. preflight  
2. deploy-check  
3. snapshot  
4. brief  
5. **judge**

O judge lê o snapshot e grava `reports/judgment_latest.json`. Para rodar apenas o judge (após snapshot/brief):

```bash
npm run shadow:judge
```

---

## Saídas esperadas

### shadow:preflight

```
--- SHADOW PREFLIGHT ---

  PASS  project directory (cwd=dashboard)
  PASS  current branch (main)
  PASS  git status (clean)
  PASS  last local commit (abc123 feat: ...)
  PASS  last origin/main commit (abc123 feat: ...)
  PASS  profile shadow_1000_structural_riskmanaged_v1 in profiles (found)
  PASS  audit route exposes structuralRiskManaged* (both present)

  Result: PASS
```

### shadow:deploy-check

```json
{
  "conclusion": "DEPLOY_OK",
  "checks": {
    "structuralRiskManagedDiagnostics_exists": true,
    "profileId_correct": true,
    "structuralRiskManagedComparison_exists": true,
    "profile_in_profileSummaries": true,
    "profile_in_rejectionCountsByProfile": true
  },
  "url": "https://.../api/shadow/audit",
  "timestamp": "2026-03-18T..."
}
```

### shadow:snapshot

Arquivo `reports/runtime_snapshot_latest.json` com:

- `economic_read_valid`, `economic_read_reason`
- `structuralRiskManagedDiagnostics`
- `structuralRiskManagedComparison_vs_shadow_1000`
- `structuralRiskManagedComparison_vs_exitrefine`
- `profileSummary`
- `metrics`
- `defenseActivation` (gate pair, capfloor, degratio, capital multiplier, early thesis failure)

### shadow:brief

Arquivo `reports/decision_brief_latest.md` com:

- Estado operacional / econômico
- Conclusão provisória / Hipótese dominante
- Alertas
- **Análise de ativação das defesas**
- **DECISÃO AGORA**
- **NÃO FAZER**
- **PRÓXIMA EVIDÊNCIA NECESSÁRIA**
- Tabela de métricas

### shadow:judge

Saída no terminal + `reports/judgment_latest.json`:

```
DEFENSIVE_LOGIC_NOT_ENGAGED
evidenceGrade: MODERATE
dominantFailureMode: defenses_not_engaged
Reason: avgCapitalMultiplier >= 0.95, earlyThesisFailure=0, capfloor=0, degratio=0...
Saved: reports/judgment_latest.json
```

O JSON inclui: `status`, `reason`, `evidenceGrade`, `dominantFailureMode`, `timestamp`, `snapshotSource`, `thresholds`.

---

## Arquivos criados

| Arquivo | Descrição |
|---------|-----------|
| `dashboard/scripts/shadow-preflight.ts` | Preflight validation |
| `dashboard/scripts/shadow-deploy-check.ts` | Deploy check contra produção |
| `dashboard/scripts/shadow-snapshot.ts` | Runtime snapshot |
| `dashboard/scripts/shadow-brief.ts` | Decision brief a partir do snapshot |
| `dashboard/scripts/shadow-full-check.ts` | Orquestrador da esteira completa |
| `dashboard/scripts/shadow-judge.ts` | Classificador de status do challenger |
| `dashboard/scripts/shadow-thresholds.ts` | Thresholds configuráveis |
| `dashboard/scripts/shadow-http.ts` | HTTP client portable (https, sem fetch global) |
| `dashboard/reports/` | Diretório de saída (snapshot, brief, judgment) |

---

## Robustez

- Não usa `jq`; lógica em TypeScript/Node
- Tratamento de erro explícito; códigos de saída 0/1
- Sem travas de pager
- Execução por comando único (`npm run shadow:full-check`)

---

## Sugestões de evolução futura

1. **CI integration**  
   Rodar `shadow:full-check` em CI e falhar o build em `DEPLOY_STALE` ou preflight FAIL.

2. **Histórico de snapshots**  
   Salvar `runtime_snapshot_YYYYMMDD_HHmm.json` além do `latest`, para análise temporal.

3. **Comparação vs baseline**  
   Incluir no brief um diff de métricas vs baseline (`shadow_1000`) quando `challengerClosed > 0`.

4. **Alertas por Slack/email**  
   Em caso de `DEPLOY_STALE` ou perda acumulada > threshold, enviar notificação.

5. **Modo local**  
   `shadow:snapshot` com `AUDIT_URL=http://localhost:3000` para validar contra dev local antes do deploy.

6. **Flags opcionais**  
   `--continue-on-deploy-stale` em full-check para não parar quando deploy estiver desatualizado.
