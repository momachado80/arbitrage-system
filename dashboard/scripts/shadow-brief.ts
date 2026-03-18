#!/usr/bin/env node
/**
 * Decision brief: reads runtime snapshot and generates executive markdown.
 *
 * Usage: npm run shadow:brief  (from dashboard/)
 * Output: reports/decision_brief_latest.md
 * Requires: reports/runtime_snapshot_latest.json (run shadow:snapshot first)
 */

import * as fs from "fs";
import * as path from "path";

const SNAPSHOT_PATH = "reports/runtime_snapshot_latest.json";
const BRIEF_PATH = "reports/decision_brief_latest.md";

interface DefenseActivation {
  structuralGateByPair: { activated: boolean; rejections: number; note: string };
  capfloor: { activated: boolean; rejections: number; note: string };
  degratio: { activated: boolean; rejections: number; note: string };
  capitalMultiplier: { activated: boolean; avgMultiplier: number; note: string };
  earlyThesisFailure: { activated: boolean; exitCount: number; note: string };
}

interface Snapshot {
  timestamp: string;
  url: string;
  economic_read_valid?: boolean;
  economic_read_reason?: string;
  structuralRiskManagedDiagnostics: Record<string, unknown> | null;
  structuralRiskManagedComparison_vs_shadow_1000: Record<string, unknown> | null;
  structuralRiskManagedComparison_vs_exitrefine: Record<string, unknown> | null;
  structuralExitKillDiagnostics?: Record<string, unknown> | null;
  structuralExitKillComparison?: Record<string, Record<string, unknown>> | null;
  structuralExitKillWindow180Diagnostics?: Record<string, unknown> | null;
  structuralExitKillWindow180Comparison?: Record<string, Record<string, unknown>> | null;
  structuralExitKillWindow180CausalAudit?: Record<string, unknown> | null;
  structuralLateExitDiagnostics?: Record<string, unknown> | null;
  structuralLateExitComparison?: Record<string, Record<string, unknown>> | null;
  profileSummary: Record<string, unknown> | null;
  exitKillProfileSummary?: Record<string, unknown> | null;
  exitKillWindow180ProfileSummary?: Record<string, unknown> | null;
  lateExitProfileSummary?: Record<string, unknown> | null;
  metrics: {
    openedTradeCount: number;
    closedTradeCount: number;
    activeTrades: number;
    avgRealizedPnL: number;
    totalRealizedPnL: number;
    avgCapitalMultiplierOpened: number;
    earlyThesisFailureExitCount: number;
    rejectedByStructuralPairCount: number;
    rejectedByCapfloorCount: number;
    rejectedByDegRatioCount: number;
  };
  defenseActivation?: DefenseActivation;
}

function num(v: unknown): number {
  return typeof v === "number" && !Number.isNaN(v) ? v : 0;
}

function run(): void {
  const cwd = path.join(__dirname, "..");
  const snapshotFile = path.join(cwd, SNAPSHOT_PATH);

  if (!fs.existsSync(snapshotFile)) {
    process.stderr.write(
      `Snapshot not found: ${snapshotFile}\nRun 'npm run shadow:snapshot' first.\n`
    );
    process.exit(1);
  }

  let snapshot: Snapshot;
  try {
    snapshot = JSON.parse(fs.readFileSync(snapshotFile, "utf-8")) as Snapshot;
  } catch (e) {
    process.stderr.write(`Invalid snapshot: ${e instanceof Error ? e.message : String(e)}\n`);
    process.exit(1);
  }

  const m = snapshot.metrics ?? {};
  const closed = num(m.closedTradeCount);
  const opened = num(m.openedTradeCount);
  const avgPnL = num(m.avgRealizedPnL);
  const totalPnL = num(m.totalRealizedPnL);
  const hasDiag = snapshot.structuralRiskManagedDiagnostics != null;

  // Operational: deploy, pipeline, data availability
  let estadoOperacional: string;
  if (!hasDiag) {
    estadoOperacional = "Indeterminado — structuralRiskManagedDiagnostics ausente. Verificar deploy.";
  } else if (opened === 0 && closed === 0) {
    estadoOperacional = "Operacional — profile ativo, sem trades abertos ainda.";
  } else {
    estadoOperacional = "Operacional — dados disponíveis.";
  }

  // Economic: only when we have closed trades
  let estadoEconomico: string;
  let conclusaoProvisoria: string;
  let hipoteseDominante: string;

  if (closed === 0) {
    estadoEconomico = "Insuficiente — challengerClosed = 0. Sem conclusão econômica possível.";
    conclusaoProvisoria =
      "Aguardar volume. Não inferir sinal econômico sem trades fechados.";
    hipoteseDominante = "N/A — volume insuficiente.";
  } else {
    const pnlSign = totalPnL >= 0 ? "positivo" : "negativo";
    estadoEconomico = `Total PnL ${pnlSign} (${totalPnL.toFixed(4)} USD). avgRealizedPnL=${avgPnL.toFixed(4)}.`;
    conclusaoProvisoria =
      totalPnL >= 0
        ? "Directional positivo — manter monitoramento."
        : "Directional negativo — investigar antes de escalar.";
    hipoteseDominante =
      totalPnL >= 0
        ? "Gate estrutural + sizing adaptativo podem estar filtrando melhor."
        : "Possível degradação ou regime desfavorável.";
  }

  // Alertas
  const alertas: string[] = [];
  if (!hasDiag) alertas.push("DEPLOY_STALE: structuralRiskManaged ausente no audit.");
  if (num(m.rejectedByStructuralPairCount) > 0 && opened === 0)
    alertas.push("Rejeições por pair sem aberturas — verificar gate de pares.");
  if (num(m.earlyThesisFailureExitCount) > 0)
    alertas.push(`Early thesis failure: ${m.earlyThesisFailureExitCount} saídas.`);
  if (closed > 0 && totalPnL < -50)
    alertas.push("Perda acumulada > 50 USD — revisar thresholds ou pausar.");
  if (alertas.length === 0) alertas.push("Nenhum alerta crítico.");

  const economicReadValid = snapshot.economic_read_valid ?? false;
  const economicReadReason = snapshot.economic_read_reason ?? "unknown";
  const defense = snapshot.defenseActivation;

  let decisaoAgora: string;
  let naoFazer: string;
  let proximaEvidencia: string;

  if (!hasDiag) {
    decisaoAgora = "Corrigir deploy. Não tomar decisão econômica.";
    naoFazer = "Não inferir performance. Não alterar thresholds.";
    proximaEvidencia = "Deploy atualizado com structural risk managed.";
  } else if (closed === 0) {
    decisaoAgora = "Aguardar. Nenhuma ação operacional.";
    naoFazer = "Não inferir sinal. Não pausar nem escalar.";
    proximaEvidencia = "Primeiro trade fechado.";
  } else if (!economicReadValid) {
    decisaoAgora = "Manter running. Não escalar nem pausar.";
    naoFazer = "Não concluir sobre hipótese. Não mudar capital.";
    proximaEvidencia =
      economicReadReason === "sample_too_small"
        ? `Atingir ${10}+ closed trades para leitura econômica válida.`
        : "Aguardar amostra mínima.";
  } else if (totalPnL < -50) {
    decisaoAgora = "Investigação. Revisar thresholds ou pausar challenger.";
    naoFazer = "Não escalar. Não ignorar perda acumulada.";
    proximaEvidencia = "Comparação vs baseline. Root cause da perda.";
  } else if (totalPnL >= 0) {
    decisaoAgora = "Manter. Acumular evidência.";
    naoFazer = "Não superpromover com amostra pequena.";
    proximaEvidencia = "Mais trades para validar direção positiva.";
  } else {
    decisaoAgora = "Monitorar. Comparar vs baseline antes de decidir.";
    naoFazer = "Não pausar prematuramente.";
    proximaEvidencia = "structuralRiskManagedComparison. Mínimo 20 closed.";
  }

  const defenseSection = defense
    ? `
---

## Análise de ativação das defesas

| Defesa | Ativada | Detalhe |
|--------|---------|---------|
| Gate estrutural (pair) | ${defense.structuralGateByPair.activated ? "Sim" : "Não"} | ${defense.structuralGateByPair.note} |
| Capfloor | ${defense.capfloor.activated ? "Sim" : "Não"} | ${defense.capfloor.note} |
| Degratio | ${defense.degratio.activated ? "Sim" : "Não"} | ${defense.degratio.note} |
| Capital multiplier | ${defense.capitalMultiplier.activated ? "Sim" : "Não"} | ${defense.capitalMultiplier.note} |
| Early thesis failure | ${defense.earlyThesisFailure.activated ? "Sim" : "Não"} | ${defense.earlyThesisFailure.note} |
`
    : "";

  const md = `# Decision Brief — shadow_1000_structural_riskmanaged_v1

**Timestamp:** ${snapshot.timestamp}
**Fonte:** ${snapshot.url}
**economic_read_valid:** ${economicReadValid}
**economic_read_reason:** ${economicReadReason}

---

## Estado operacional

${estadoOperacional}

---

## Estado econômico

${estadoEconomico}

---

## Conclusão provisória

${conclusaoProvisoria}

---

## Hipótese dominante

${hipoteseDominante}

---

## Alertas

${alertas.map((a) => `- ${a}`).join("\n")}
${defenseSection}
---

## DECISÃO AGORA

${decisaoAgora}

---

## NÃO FAZER

${naoFazer}

---

## PRÓXIMA EVIDÊNCIA NECESSÁRIA

${proximaEvidencia}

---

## Métricas resumidas

| Métrica | Valor |
|---------|-------|
| openedTradeCount | ${m.openedTradeCount ?? 0} |
| closedTradeCount | ${m.closedTradeCount ?? 0} |
| avgRealizedPnL | ${(m.avgRealizedPnL ?? 0).toFixed(4)} |
| totalRealizedPnL | ${(m.totalRealizedPnL ?? 0).toFixed(4)} |
| avgCapitalMultiplierOpened | ${(m.avgCapitalMultiplierOpened ?? 0).toFixed(2)} |
| earlyThesisFailureExitCount | ${m.earlyThesisFailureExitCount ?? 0} |
| rejectedByStructuralPairCount | ${m.rejectedByStructuralPairCount ?? 0} |
| rejectedByCapfloorCount | ${m.rejectedByCapfloorCount ?? 0} |
| rejectedByDegRatioCount | ${m.rejectedByDegRatioCount ?? 0} |
${
  snapshot.structuralExitKillDiagnostics != null
    ? `
---

## Exit Kill Challenger (shadow_1000_structural_exitkill_v1)

| Métrica | Valor |
|---------|-------|
| closedTradeCount | ${num(snapshot.structuralExitKillDiagnostics.closedTradeCount)} |
| earlyKillExitCount | ${num(snapshot.structuralExitKillDiagnostics.earlyKillExitCount)} |
| avgRealizedPnL | ${num(snapshot.structuralExitKillDiagnostics.avgRealizedPnL).toFixed(4)} |
| totalRealizedPnL | ${num(snapshot.structuralExitKillDiagnostics.totalRealizedPnL).toFixed(4)} |
| avgHoldingMsEarlyKill | ${num(snapshot.structuralExitKillDiagnostics.avgHoldingMsEarlyKill).toFixed(0)} ms |
| killReasonCounts | ${JSON.stringify(snapshot.structuralExitKillDiagnostics.killReasonCounts ?? {})} |
`
    : ""
}
${
  snapshot.structuralExitKillWindow180Diagnostics != null
    ? `
---

## Exit Kill Window 180 (shadow_1000_structural_exitkill_window180_v1)

| Métrica | Valor |
|---------|-------|
| closedTradeCount | ${num(snapshot.structuralExitKillWindow180Diagnostics.closedTradeCount)} |
| earlyKillExitCount | ${num(snapshot.structuralExitKillWindow180Diagnostics.earlyKillExitCount)} |
| avgRealizedPnL | ${num(snapshot.structuralExitKillWindow180Diagnostics.avgRealizedPnL).toFixed(4)} |
| totalRealizedPnL | ${num(snapshot.structuralExitKillWindow180Diagnostics.totalRealizedPnL).toFixed(4)} |
| avgHoldingTimeMs | ${num(snapshot.structuralExitKillWindow180Diagnostics.avgHoldingTimeMs).toFixed(0)} ms |
| avgHoldingMsEarlyKill | ${num(snapshot.structuralExitKillWindow180Diagnostics.avgHoldingMsEarlyKill).toFixed(0)} ms |
| killReasonCounts | ${JSON.stringify(snapshot.structuralExitKillWindow180Diagnostics.killReasonCounts ?? {})} |
${
  snapshot.structuralExitKillWindow180CausalAudit != null
    ? `| closedInKillWindow | ${num(snapshot.structuralExitKillWindow180CausalAudit.closedInKillWindow)} |
| closedOutsideKillWindow | ${num(snapshot.structuralExitKillWindow180CausalAudit.closedOutsideKillWindow)} |
| killWindowMs | ${num(snapshot.structuralExitKillWindow180CausalAudit.killWindowMs)} |`
    : ""
}
`
    : ""
}
${
  snapshot.structuralLateExitDiagnostics != null
    ? `
---

## Late Exit Non-Reversion (shadow_1000_structural_lateexit_nonreversion_v1)

| Métrica | Valor |
|---------|-------|
| closedTradeCount | ${num(snapshot.structuralLateExitDiagnostics.closedTradeCount)} |
| lateExitTriggeredCount | ${num(snapshot.structuralLateExitDiagnostics.lateExitTriggeredCount)} |
| avgRealizedPnL | ${num(snapshot.structuralLateExitDiagnostics.avgRealizedPnL).toFixed(4)} |
| totalRealizedPnL | ${num(snapshot.structuralLateExitDiagnostics.totalRealizedPnL).toFixed(4)} |
| avgHoldingTimeMs | ${num(snapshot.structuralLateExitDiagnostics.avgHoldingTimeMs).toFixed(0)} ms |
| avgHoldingMsLateExit | ${num(snapshot.structuralLateExitDiagnostics.avgHoldingMsLateExit).toFixed(0)} ms |
| lateExitReasonCounts | ${JSON.stringify(snapshot.structuralLateExitDiagnostics.lateExitReasonCounts ?? {})} |
`
    : ""
}
`;

  const outDir = path.join(cwd, "reports");
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  const outPath = path.join(outDir, "decision_brief_latest.md");
  fs.writeFileSync(outPath, md, "utf-8");
  process.stdout.write(`Brief saved: ${outPath}\n`);
}

run();
