#!/usr/bin/env node
/**
 * Shadow judge: classifies challenger from snapshot into a closed status enum.
 * Conservative — never over-promotes with small sample.
 *
 * Usage: npm run shadow:judge  (from dashboard/)
 * Requires: reports/runtime_snapshot_latest.json (run shadow:snapshot first)
 */

import * as fs from "fs";
import * as path from "path";
import { THRESHOLDS } from "./shadow-thresholds";

export type ChallengerStatus =
  | "NOT_READABLE_YET"
  | "SAMPLE_TOO_SMALL"
  | "OPERABLE_BUT_UNPROVEN"
  | "WORSE_THAN_BASELINE"
  | "LESS_BAD_THAN_BASELINE"
  | "DEFENSIVE_LOGIC_NOT_ENGAGED"
  | "PROMISING_BUT_EARLY";

export type EvidenceGrade = "NONE" | "WEAK" | "MODERATE" | "STRONG";

export type DominantFailureMode =
  | "deploy_not_live"
  | "no_closed_sample"
  | "structural_gate_too_narrow"
  | "defenses_not_engaged"
  | "worse_than_baseline"
  | "insufficient_comparison_sample"
  | "no_clear_failure_mode";

const SNAPSHOT_PATH = "reports/runtime_snapshot_latest.json";

function num(v: unknown): number {
  return typeof v === "number" && !Number.isNaN(v) ? v : 0;
}

interface Snapshot {
  economic_read_valid?: boolean;
  economic_read_reason?: string;
  structuralRiskManagedDiagnostics: Record<string, unknown> | null;
  structuralRiskManagedComparison_vs_shadow_1000: Record<string, unknown> | null;
  profileSummary: Record<string, unknown> | null;
  structuralExitKillDiagnostics?: Record<string, unknown> | null;
  structuralExitKillComparison?: Record<string, Record<string, unknown>> | null;
  structuralExitKillWindow180Diagnostics?: Record<string, unknown> | null;
  structuralExitKillWindow180Comparison?: Record<string, Record<string, unknown>> | null;
  exitKillProfileSummary?: Record<string, unknown> | null;
  exitKillWindow180ProfileSummary?: Record<string, unknown> | null;
  metrics: Record<string, unknown>;
  defenseActivation?: Record<string, { activated: boolean; rejections?: number; avgMultiplier?: number }>;
}

interface Judgment {
  status: ChallengerStatus;
  reason: string;
  evidenceGrade: EvidenceGrade;
  dominantFailureMode: DominantFailureMode;
  timestamp: string;
  snapshotSource: string;
  thresholds: {
    minClosedReadable: number;
    minClosedCompare: number;
    minImprovementPct: number;
  };
}

/** Adaptive defesas (capfloor, degratio, multiplier, early exit) — não inclui structural gate por ser filtro de universo */
function adaptiveDefensesNotEngaged(snapshot: Snapshot): boolean {
  const diag = snapshot.structuralRiskManagedDiagnostics as Record<string, unknown> | undefined;
  const m = snapshot.metrics ?? {};
  const avgMult = num(diag?.avgCapitalMultiplierOpened ?? m.avgCapitalMultiplierOpened);
  const earlyCount = num(diag?.earlyThesisFailureExitCount ?? m.earlyThesisFailureExitCount);
  const capRej = num(diag?.rejectedByCapfloorCount ?? m.rejectedByCapfloorCount);
  const degRej = num(diag?.rejectedByDegRatioCount ?? m.rejectedByDegRatioCount);
  return avgMult >= 0.95 && earlyCount === 0 && capRej === 0 && degRej === 0;
}

/** Pelo menos uma defesa adaptativa ou estrutural claramente ativada */
function atLeastOneDefenseActivated(snapshot: Snapshot): boolean {
  const d = snapshot.defenseActivation ?? {};
  return (
    (d.structuralGateByPair?.activated ?? false) ||
    (d.capfloor?.activated ?? false) ||
    (d.degratio?.activated ?? false) ||
    (d.capitalMultiplier?.activated ?? false) ||
    (d.earlyThesisFailure?.activated ?? false)
  );
}

/** Gate estrutural rejeitou a maioria das opps — possível gate muito estreito */
function structuralGateTooNarrow(snapshot: Snapshot): boolean {
  const diag = snapshot.structuralRiskManagedDiagnostics as Record<string, unknown> | undefined;
  const evalCount = num(diag?.evaluatedOpportunityCount);
  const pairRej = num(diag?.rejectedByStructuralPairCount);
  if (evalCount < 100) return false;
  const pairRejRatio = pairRej / evalCount;
  return pairRejRatio >= 0.9;
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
  const totalPnL = num(m.totalRealizedPnL);
  const challengerAvg = num(
    (snapshot.structuralRiskManagedComparison_vs_shadow_1000 as Record<string, unknown> | null)
      ?.challengerAvgRealizedPnL
  );
  const hasDiag = snapshot.structuralRiskManagedDiagnostics != null;
  const compVs1000 = snapshot.structuralRiskManagedComparison_vs_shadow_1000 as Record<string, unknown> | null;
  const economicReadValid = snapshot.economic_read_valid ?? false;
  const economicReadReason = snapshot.economic_read_reason ?? "unknown";

  const baselineAvg = num(compVs1000?.baselineAvgRealizedPnL);
  const challengerClosedFromComp = num(compVs1000?.challengerClosed);
  const comparisonClosed = Math.min(closed, challengerClosedFromComp || closed);

  let status: ChallengerStatus;
  let reason: string;
  let dominantFailureMode: DominantFailureMode;
  let evidenceGrade: EvidenceGrade;

  // --- evidenceGrade (conservador) ---
  if (!hasDiag || closed === 0) {
    evidenceGrade = "NONE";
  } else if (closed < THRESHOLDS.minimumClosedForReadableEconomics) {
    evidenceGrade = "NONE";
  } else if (closed < THRESHOLDS.minimumClosedForBaselineComparison) {
    evidenceGrade = "WEAK";
  } else if (comparisonClosed >= THRESHOLDS.minimumClosedForBaselineComparison) {
    evidenceGrade = "MODERATE"; // pode subir para STRONG com mais critérios no futuro
  } else {
    evidenceGrade = "WEAK";
  }
  if (evidenceGrade === "MODERATE" && closed >= 50) evidenceGrade = "STRONG";

  // --- 1. NOT_READABLE_YET: deploy_stale OU challenger_closed_zero ---
  if (!hasDiag) {
    status = "NOT_READABLE_YET";
    reason = "deploy_stale — structuralRiskManagedDiagnostics ausente.";
    dominantFailureMode = "deploy_not_live";
  } else if (closed === 0) {
    status = "NOT_READABLE_YET";
    reason = "challenger_closed_zero — nenhum trade fechado.";
    dominantFailureMode = "no_closed_sample";
  }
  // --- 2. SAMPLE_TOO_SMALL: closed > 0 e closed < MIN_READABLE ---
  else if (closed > 0 && closed < THRESHOLDS.minimumClosedForReadableEconomics) {
    status = "SAMPLE_TOO_SMALL";
    reason = `closed=${closed} (0 < x < ${THRESHOLDS.minimumClosedForReadableEconomics}). Amostra insuficiente para leitura econômica.`;
    dominantFailureMode = structuralGateTooNarrow(snapshot) ? "structural_gate_too_narrow" : "insufficient_comparison_sample";
  }
  // --- 3+ Economic read valid: avaliar defesas e comparação ---
  else {
    const defensesNotEngaged = adaptiveDefensesNotEngaged(snapshot);
    const challengerBadOrInconclusive = challengerAvg < 0 || (challengerAvg >= 0 && closed < THRESHOLDS.minimumClosedForBaselineComparison);

    // DEFENSIVE_LOGIC_NOT_ENGAGED: adaptive defesas não acionaram E challenger ruim/inconclusivo
    if (defensesNotEngaged && challengerBadOrInconclusive) {
      status = "DEFENSIVE_LOGIC_NOT_ENGAGED";
      reason =
        "avgCapitalMultiplier >= 0.95, earlyThesisFailure=0, capfloor=0, degratio=0. Defesas adaptativas não morderam e challenger ruim/inconclusivo.";
      dominantFailureMode = "defenses_not_engaged";
    }
    // Comparação disponível (closed >= MIN_COMPARE)
    else if (closed >= THRESHOLDS.minimumClosedForBaselineComparison) {
      if (baselineAvg >= 0 && challengerAvg < 0) {
        status = "WORSE_THAN_BASELINE";
        reason = "Baseline positivo, challenger negativo.";
        dominantFailureMode = "worse_than_baseline";
      } else if (baselineAvg < 0 && challengerAvg >= 0) {
        status = "LESS_BAD_THAN_BASELINE";
        reason = "Challenger positivo vs baseline negativo.";
        dominantFailureMode = "no_clear_failure_mode";
      } else if (baselineAvg < 0 && challengerAvg < 0) {
        const improvement = baselineAvg === 0 ? 0 : (challengerAvg - baselineAvg) / Math.abs(baselineAvg);
        if (improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
          status = "LESS_BAD_THAN_BASELINE";
          reason = `Challenger ${(improvement * 100).toFixed(1)}% menos ruim que baseline.`;
          dominantFailureMode = "no_clear_failure_mode";
        } else if (improvement <= -THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
          status = "WORSE_THAN_BASELINE";
          reason = `Challenger ${(-improvement * 100).toFixed(1)}% pior que baseline.`;
          dominantFailureMode = "worse_than_baseline";
        } else {
          status = "OPERABLE_BUT_UNPROVEN";
          reason = `Diferença vs baseline dentro da banda de ruído (${(improvement * 100).toFixed(1)}%). Comparação inconclusiva.`;
          dominantFailureMode = "no_clear_failure_mode";
        }
      } else if (challengerAvg >= 0) {
        status = "OPERABLE_BUT_UNPROVEN";
        reason = "Challenger positivo com amostra suficiente. Manter monitoramento.";
        dominantFailureMode = "no_clear_failure_mode";
      } else {
        status = "OPERABLE_BUT_UNPROVEN";
        reason = "Comparação inconclusiva.";
        dominantFailureMode = "no_clear_failure_mode";
      }
    }
    // closed >= MIN_READABLE mas < MIN_COMPARE (sample abaixo do ideal para comparação)
    else {
      const improvement =
        baselineAvg === 0 ? 0 : (challengerAvg - baselineAvg) / Math.abs(baselineAvg);
      const challengerBetterByThreshold =
        (challengerAvg >= 0 && baselineAvg < 0) ||
        (baselineAvg < 0 && challengerAvg < 0 && improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad);
      const hasDefense = atLeastOneDefenseActivated(snapshot);

      // PROMISING_BUT_EARLY: raro — sample abaixo do ideal, challenger melhor que baseline por threshold, ≥1 defesa ativada
      if (challengerBetterByThreshold && hasDefense) {
        status = "PROMISING_BUT_EARLY";
        reason = `Sample ainda abaixo do ideal (closed=${closed} < ${THRESHOLDS.minimumClosedForBaselineComparison}). Challenger melhor que baseline por threshold. ≥1 defesa ativada.`;
        dominantFailureMode = "no_clear_failure_mode";
      } else {
        status = "OPERABLE_BUT_UNPROVEN";
        reason = `economic_read_valid true, mas closed=${closed} < ${THRESHOLDS.minimumClosedForBaselineComparison}. Comparação insuficiente ou sem superioridade clara.`;
        dominantFailureMode = closed < THRESHOLDS.minimumClosedForBaselineComparison
          ? "insufficient_comparison_sample"
          : "no_clear_failure_mode";
      }
    }
  }

  const judgment: Judgment = {
    status,
    reason,
    evidenceGrade,
    dominantFailureMode,
    timestamp: new Date().toISOString(),
    snapshotSource: snapshotFile,
    thresholds: {
      minClosedReadable: THRESHOLDS.minimumClosedForReadableEconomics,
      minClosedCompare: THRESHOLDS.minimumClosedForBaselineComparison,
      minImprovementPct: THRESHOLDS.minimumRelativeImprovementToCallLessBad,
    },
  };

  const outDir = path.join(cwd, "reports");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "judgment_latest.json");
  fs.writeFileSync(outPath, JSON.stringify(judgment, null, 2), "utf-8");

  process.stdout.write(`[structural_riskmanaged] ${status}\n`);
  process.stdout.write(`evidenceGrade: ${evidenceGrade}\n`);
  process.stdout.write(`dominantFailureMode: ${dominantFailureMode}\n`);
  process.stdout.write(`Reason: ${reason}\n`);
  process.stdout.write(`Saved: ${outPath}\n`);

  // Exit kill challenger: conservative classification when data exists
  const exitKillDiag = snapshot.structuralExitKillDiagnostics as Record<string, unknown> | null | undefined;
  const exitKillComp = snapshot.structuralExitKillComparison as Record<string, Record<string, unknown>> | null | undefined;
  if (exitKillDiag != null && exitKillComp != null) {
    const ekClosed = num(exitKillDiag.closedTradeCount);
    const compVs1000 = exitKillComp["shadow_1000"];
    const compVsRiskManaged = exitKillComp["shadow_1000_structural_riskmanaged_v1"];
    const ekChallengerAvg = num(exitKillDiag.avgRealizedPnL);
    const ekEarlyKill = num(exitKillDiag.earlyKillExitCount);
    const ekDefenseActivated = ekEarlyKill > 0;

    let ekStatus: ChallengerStatus;
    let ekReason: string;
    let ekEvidenceGrade: EvidenceGrade = ekClosed >= THRESHOLDS.minimumClosedForBaselineComparison ? "MODERATE" : ekClosed >= THRESHOLDS.minimumClosedForReadableEconomics ? "WEAK" : "NONE";
    if (ekClosed >= 50) ekEvidenceGrade = "STRONG";

    if (ekClosed === 0) {
      ekStatus = "NOT_READABLE_YET";
      ekReason = "challenger_closed_zero — nenhum trade fechado.";
    } else if (ekClosed < THRESHOLDS.minimumClosedForReadableEconomics) {
      ekStatus = "SAMPLE_TOO_SMALL";
      ekReason = `closed=${ekClosed}. Amostra insuficiente.`;
    } else if (!ekDefenseActivated && ekChallengerAvg < 0) {
      ekStatus = "DEFENSIVE_LOGIC_NOT_ENGAGED";
      ekReason = "earlyKillExitCount=0. Saída adaptativa não acionou e challenger negativo.";
    } else if (compVs1000 != null && ekClosed >= THRESHOLDS.minimumClosedForBaselineComparison) {
      const baselineAvg = num(compVs1000.baselineAvgRealizedPnL);
      const improvement = baselineAvg === 0 ? 0 : (ekChallengerAvg - baselineAvg) / Math.abs(baselineAvg);
      if (baselineAvg >= 0 && ekChallengerAvg < 0) {
        ekStatus = "WORSE_THAN_BASELINE";
        ekReason = "Baseline positivo, exit-kill negativo.";
      } else if (baselineAvg < 0 && ekChallengerAvg >= 0) {
        ekStatus = "LESS_BAD_THAN_BASELINE";
        ekReason = "Exit-kill positivo vs baseline negativo.";
      } else if (improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        ekStatus = "LESS_BAD_THAN_BASELINE";
        ekReason = `Exit-kill ${(improvement * 100).toFixed(1)}% menos ruim que baseline.`;
      } else if (improvement <= -THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        ekStatus = "WORSE_THAN_BASELINE";
        ekReason = `Exit-kill ${(-improvement * 100).toFixed(1)}% pior que baseline.`;
      } else {
        ekStatus = "OPERABLE_BUT_UNPROVEN";
        ekReason = "Diferença vs baseline dentro da banda de ruído.";
      }
    } else if (compVsRiskManaged != null && ekClosed >= THRESHOLDS.minimumClosedForBaselineComparison) {
      const rmAvg = num(compVsRiskManaged.baselineAvgRealizedPnL);
      const improvement = rmAvg === 0 ? 0 : (ekChallengerAvg - rmAvg) / Math.abs(rmAvg);
      if (improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        ekStatus = "LESS_BAD_THAN_BASELINE";
        ekReason = `Exit-kill ${(improvement * 100).toFixed(1)}% menos ruim que structural_riskmanaged.`;
      } else if (improvement <= -THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        ekStatus = "WORSE_THAN_BASELINE";
        ekReason = `Exit-kill ${(-improvement * 100).toFixed(1)}% pior que structural_riskmanaged.`;
      } else {
        ekStatus = "OPERABLE_BUT_UNPROVEN";
        ekReason = "Comparação vs riskmanaged inconclusiva.";
      }
    } else {
      ekStatus = "OPERABLE_BUT_UNPROVEN";
      ekReason = "Aguardar amostra para comparação ou defesa mais acionada.";
    }

    const ekJudgment: Judgment = {
      status: ekStatus,
      reason: ekReason,
      evidenceGrade: ekEvidenceGrade,
      dominantFailureMode: ekStatus === "WORSE_THAN_BASELINE" ? "worse_than_baseline" : ekStatus === "DEFENSIVE_LOGIC_NOT_ENGAGED" ? "defenses_not_engaged" : "no_clear_failure_mode",
      timestamp: new Date().toISOString(),
      snapshotSource: snapshotFile,
      thresholds: {
        minClosedReadable: THRESHOLDS.minimumClosedForReadableEconomics,
        minClosedCompare: THRESHOLDS.minimumClosedForBaselineComparison,
        minImprovementPct: THRESHOLDS.minimumRelativeImprovementToCallLessBad,
      },
    };
    const ekOutPath = path.join(outDir, "judgment_exitkill_latest.json");
    fs.writeFileSync(ekOutPath, JSON.stringify(ekJudgment, null, 2), "utf-8");
    process.stdout.write(`\n[exitkill] ${ekStatus}\n`);
    process.stdout.write(`evidenceGrade: ${ekEvidenceGrade}\n`);
    process.stdout.write(`Reason: ${ekReason}\n`);
    process.stdout.write(`Saved: ${ekOutPath}\n`);
  }

  // Exit kill window 180: mesma lógica, comparação vs exitkill_v1
  const w180Diag = snapshot.structuralExitKillWindow180Diagnostics as Record<string, unknown> | null | undefined;
  const w180Comp = snapshot.structuralExitKillWindow180Comparison as Record<string, Record<string, unknown>> | null | undefined;
  if (w180Diag != null && w180Comp != null) {
    const w180Closed = num(w180Diag.closedTradeCount);
    const compVs1000 = w180Comp["shadow_1000"];
    const compVsExitkill = w180Comp["shadow_1000_structural_exitkill_v1"];
    const w180ChallengerAvg = num(w180Diag.avgRealizedPnL);
    const w180EarlyKill = num(w180Diag.earlyKillExitCount);
    const w180DefenseActivated = w180EarlyKill > 0;

    let w180Status: ChallengerStatus;
    let w180Reason: string;
    let w180EvidenceGrade: EvidenceGrade = w180Closed >= THRESHOLDS.minimumClosedForBaselineComparison ? "MODERATE" : w180Closed >= THRESHOLDS.minimumClosedForReadableEconomics ? "WEAK" : "NONE";
    if (w180Closed >= 50) w180EvidenceGrade = "STRONG";

    if (w180Closed === 0) {
      w180Status = "NOT_READABLE_YET";
      w180Reason = "challenger_closed_zero — nenhum trade fechado.";
    } else if (w180Closed < THRESHOLDS.minimumClosedForReadableEconomics) {
      w180Status = "SAMPLE_TOO_SMALL";
      w180Reason = `closed=${w180Closed}. Amostra insuficiente.`;
    } else if (!w180DefenseActivated && w180ChallengerAvg < 0) {
      w180Status = "DEFENSIVE_LOGIC_NOT_ENGAGED";
      w180Reason = "earlyKillExitCount=0. Kill window 180s não acionou ainda e challenger negativo.";
    } else if (compVs1000 != null && w180Closed >= THRESHOLDS.minimumClosedForBaselineComparison) {
      const baselineAvg = num(compVs1000.baselineAvgRealizedPnL);
      const improvement = baselineAvg === 0 ? 0 : (w180ChallengerAvg - baselineAvg) / Math.abs(baselineAvg);
      if (baselineAvg >= 0 && w180ChallengerAvg < 0) {
        w180Status = "WORSE_THAN_BASELINE";
        w180Reason = "Baseline positivo, window180 negativo.";
      } else if (baselineAvg < 0 && w180ChallengerAvg >= 0) {
        w180Status = "LESS_BAD_THAN_BASELINE";
        w180Reason = "Window180 positivo vs baseline negativo.";
      } else if (improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        w180Status = "LESS_BAD_THAN_BASELINE";
        w180Reason = `Window180 ${(improvement * 100).toFixed(1)}% menos ruim que baseline.`;
      } else if (improvement <= -THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        w180Status = "WORSE_THAN_BASELINE";
        w180Reason = `Window180 ${(-improvement * 100).toFixed(1)}% pior que baseline.`;
      } else {
        w180Status = "OPERABLE_BUT_UNPROVEN";
        w180Reason = "Diferença vs baseline dentro da banda de ruído.";
      }
    } else if (compVsExitkill != null && w180Closed >= THRESHOLDS.minimumClosedForBaselineComparison) {
      const ekAvg = num(compVsExitkill.baselineAvgRealizedPnL);
      const improvement = ekAvg === 0 ? 0 : (w180ChallengerAvg - ekAvg) / Math.abs(ekAvg);
      if (improvement >= THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        w180Status = "LESS_BAD_THAN_BASELINE";
        w180Reason = `Window180 ${(improvement * 100).toFixed(1)}% menos ruim que exitkill_v1.`;
      } else if (improvement <= -THRESHOLDS.minimumRelativeImprovementToCallLessBad) {
        w180Status = "WORSE_THAN_BASELINE";
        w180Reason = `Window180 ${(-improvement * 100).toFixed(1)}% pior que exitkill_v1.`;
      } else {
        w180Status = "OPERABLE_BUT_UNPROVEN";
        w180Reason = "Comparação vs exitkill_v1 inconclusiva.";
      }
    } else {
      w180Status = "OPERABLE_BUT_UNPROVEN";
      w180Reason = "Aguardar amostra para comparação. Primeiro objetivo: verificar se earlyKillExitCount > 0.";
    }

    const w180Judgment: Judgment = {
      status: w180Status,
      reason: w180Reason,
      evidenceGrade: w180EvidenceGrade,
      dominantFailureMode: w180Status === "WORSE_THAN_BASELINE" ? "worse_than_baseline" : w180Status === "DEFENSIVE_LOGIC_NOT_ENGAGED" ? "defenses_not_engaged" : "no_clear_failure_mode",
      timestamp: new Date().toISOString(),
      snapshotSource: snapshotFile,
      thresholds: {
        minClosedReadable: THRESHOLDS.minimumClosedForReadableEconomics,
        minClosedCompare: THRESHOLDS.minimumClosedForBaselineComparison,
        minImprovementPct: THRESHOLDS.minimumRelativeImprovementToCallLessBad,
      },
    };
    const w180OutPath = path.join(outDir, "judgment_exitkill_window180_latest.json");
    fs.writeFileSync(w180OutPath, JSON.stringify(w180Judgment, null, 2), "utf-8");
    process.stdout.write(`\n[exitkill_window180] ${w180Status}\n`);
    process.stdout.write(`evidenceGrade: ${w180EvidenceGrade}\n`);
    process.stdout.write(`earlyKillExitCount: ${w180EarlyKill}\n`);
    process.stdout.write(`Reason: ${w180Reason}\n`);
    process.stdout.write(`Saved: ${w180OutPath}\n`);
  }
}

run();
