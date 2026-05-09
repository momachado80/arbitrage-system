/**
 * Resumo operacional consolidado da trilha catalog → economics → execution → minimal paper.
 * Só agrega campos já calculados pelos digests; não recalcula regras nem thresholds.
 */

import { ensureCatalogPocketProbe, buildCatalogPocketDigest } from "./catalogPocketProbe";
import { ensurePocketEconomicsProbe, buildPocketEconomicsDigest } from "./pocketEconomicsProbe";
import { ensurePocketExecutionProbe, buildPocketExecutionDigest } from "./pocketExecutionProbe";
import {
  ensureMinimalPaperExecutionProbe,
  buildMinimalPaperExecutionDigest,
} from "./minimalPaperExecutionProbe";
import { buildMicroEdgeSummaryLine } from "./minimalPaperMicroEdgeAssessment";
import { buildPaperPersistedStateHygieneHealth } from "./paperProbePersistedState";
import {
  processLadderHistoryForDigest,
  type LadderHistoryDigest,
  type LadderTrajectoryAssessmentDigest,
} from "./systemLadderHistory";
import {
  ensureMomentumSnipingProbe,
  buildMomentumSnipingDigest,
  type MomentumSnipingDigest,
} from "./momentumSnipingProbe";
import {
  buildRankedSignalSummaryLine,
  type RankedSignalVerdict,
} from "./momentumRankedEventAssessment";
import {
  buildRobustnessSummaryLine,
  type TopSliceRobustnessVerdict,
} from "./momentumTopSliceRobustness";
import {
  buildSelectionSummaryLine,
  type SelectionAssessmentVerdict,
} from "./momentumTopSliceSelection";
import {
  buildOperationalizationSummaryLine,
  type OperationalizationVerdict,
} from "./momentumOperationalization";
import {
  buildOperationalizationV2SummaryLine,
  type OperationalizationV2Verdict,
} from "./momentumOperationalizationV2";
import {
  buildOpsRobustnessSummaryLine,
  type OpsRobustnessVerdict,
} from "./operationalizationRobustness";
import {
  buildPromotionReadinessSummaryLine,
  type PromotionConfidenceTier,
} from "./operationalizationPromotionReadiness";
import { buildProgressSummaryLine } from "./promotionProgressTracker";
import {
  buildRealisticPaperSummaryLine,
  type RealisticPaperVerdict,
} from "./realisticPaperExecutionAssessment";
import type { ExecutionSurvivabilityVerdict } from "./executionSurvivabilitySegmentation";
import type { SegmentedPaperPreparationVerdict } from "./segmentedPaperTestPreparation";
import type { SegmentedPaperExecutionVerdict } from "./segmentedPaperExecutionAssessment";
import type { SegmentedWave2ExecutionVerdict } from "./segmentedPaperExecutionWave2Assessment";

const TARGET_FAMILY_KEY = "other:price_above:>3m" as const;

/** Janelas só para leitura operacional do agregado; não são gates dos probes. */
const TEMPORAL_SPREAD_STALE_MS = 48 * 3600_000;
const TEMPORAL_RECENT_MS = 24 * 3600_000;
const TEMPORAL_AGING_MS = 7 * 24 * 3600_000;

export type LayerDataFreshness = "no_observation" | "recent" | "aging" | "stale";

export type OverallTemporalConsistencyVerdict = "consistent" | "partially_stale" | "not_comparable_yet";

function take<T>(arr: T[] | undefined, n: number): T[] {
  if (!arr || arr.length === 0) return [];
  return arr.slice(0, n);
}

function parseIsoMs(iso: string | null | undefined): number | null {
  if (typeof iso !== "string" || iso.length < 10) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

function dataFreshnessFromAge(ageMs: number | null): LayerDataFreshness {
  if (ageMs === null) return "no_observation";
  if (ageMs <= TEMPORAL_RECENT_MS) return "recent";
  if (ageMs <= TEMPORAL_AGING_MS) return "aging";
  return "stale";
}

function layerTemporalFields(
  lastSuccessfulScanAt: string | null,
  nowMs: number,
): {
  lastMeaningfulObservationAt: string | null;
  dataFreshness: LayerDataFreshness;
  hasComparableLayer: boolean;
} {
  const obsMs = parseIsoMs(lastSuccessfulScanAt);
  const hasComparableLayer = obsMs !== null;
  const ageMs = obsMs !== null ? Math.max(0, nowMs - obsMs) : null;
  return {
    lastMeaningfulObservationAt: lastSuccessfulScanAt,
    dataFreshness: dataFreshnessFromAge(ageMs),
    hasComparableLayer,
  };
}

function buildTemporalConsistency(args: {
  nowMs: number;
  catalog: { lastSuccessfulScanAt: string | null };
  econ: { lastSuccessfulScanAt: string | null };
  exec: { lastSuccessfulScanAt: string | null };
  minimal: { lastSuccessfulScanAt: string | null };
}): {
  hasComparableCatalogPocketLayer: boolean;
  hasComparablePocketEconomicsLayer: boolean;
  hasComparablePocketExecutionLayer: boolean;
  hasComparableMinimalPaperLayer: boolean;
  overallTemporalConsistencyVerdict: OverallTemporalConsistencyVerdict;
  consistencyNotes: string[];
  comparableLayerTimeSpreadMs: number | null;
} {
  const c = layerTemporalFields(args.catalog.lastSuccessfulScanAt, args.nowMs).hasComparableLayer;
  const e = layerTemporalFields(args.econ.lastSuccessfulScanAt, args.nowMs).hasComparableLayer;
  const x = layerTemporalFields(args.exec.lastSuccessfulScanAt, args.nowMs).hasComparableLayer;
  const m = layerTemporalFields(args.minimal.lastSuccessfulScanAt, args.nowMs).hasComparableLayer;

  const notes: string[] = [];
  const ts = {
    catalog: parseIsoMs(args.catalog.lastSuccessfulScanAt),
    econ: parseIsoMs(args.econ.lastSuccessfulScanAt),
    exec: parseIsoMs(args.exec.lastSuccessfulScanAt),
    minimal: parseIsoMs(args.minimal.lastSuccessfulScanAt),
  };

  if (!c && !e && !x && !m) {
    notes.push("Nenhuma camada tem lastSuccessfulScanAt — aguardar primeiro scan bem-sucedido ou reidratação com histórico.");
    return {
      hasComparableCatalogPocketLayer: false,
      hasComparablePocketEconomicsLayer: false,
      hasComparablePocketExecutionLayer: false,
      hasComparableMinimalPaperLayer: false,
      overallTemporalConsistencyVerdict: "not_comparable_yet",
      consistencyNotes: notes,
      comparableLayerTimeSpreadMs: null,
    };
  }

  if (e && !c) {
    notes.push(
      "pocket-economics tem observação bem-sucedida mas catalog-pocket não; promotionLadder (ex.: recurrentPocket) pode não estar alinhado no tempo com a base catalog.",
    );
  }
  if (x && !e) {
    notes.push(
      "pocket-execution tem observação bem-sucedida mas pocket-economics não; leitura da escada pode estar desfasada.",
    );
  }
  if (m && !x) {
    notes.push(
      "minimal-paper-execution tem observação bem-sucedida mas pocket-execution não; contexto temporal incompleto.",
    );
  }

  if (e && !c) {
    return {
      hasComparableCatalogPocketLayer: c,
      hasComparablePocketEconomicsLayer: e,
      hasComparablePocketExecutionLayer: x,
      hasComparableMinimalPaperLayer: m,
      overallTemporalConsistencyVerdict: "not_comparable_yet",
      consistencyNotes: notes,
      comparableLayerTimeSpreadMs: null,
    };
  }
  if (x && !e) {
    return {
      hasComparableCatalogPocketLayer: c,
      hasComparablePocketEconomicsLayer: e,
      hasComparablePocketExecutionLayer: x,
      hasComparableMinimalPaperLayer: m,
      overallTemporalConsistencyVerdict: "not_comparable_yet",
      consistencyNotes: notes,
      comparableLayerTimeSpreadMs: null,
    };
  }
  if (m && !x) {
    return {
      hasComparableCatalogPocketLayer: c,
      hasComparablePocketEconomicsLayer: e,
      hasComparablePocketExecutionLayer: x,
      hasComparableMinimalPaperLayer: m,
      overallTemporalConsistencyVerdict: "not_comparable_yet",
      consistencyNotes: notes,
      comparableLayerTimeSpreadMs: null,
    };
  }

  const observed = [ts.catalog, ts.econ, ts.exec, ts.minimal].filter((t): t is number => t !== null);
  let spread: number | null = null;
  if (observed.length >= 2) {
    spread = Math.max(...observed) - Math.min(...observed);
    if (spread > TEMPORAL_SPREAD_STALE_MS) {
      notes.push(
        `Desvio temporal entre camadas com scan bem-sucedido: ~${Math.round(spread / 3600000)}h (> ${TEMPORAL_SPREAD_STALE_MS / 3600000}h) — interpretar promotionLadder com cautela.`,
      );
      return {
        hasComparableCatalogPocketLayer: c,
        hasComparablePocketEconomicsLayer: e,
        hasComparablePocketExecutionLayer: x,
        hasComparableMinimalPaperLayer: m,
        overallTemporalConsistencyVerdict: "partially_stale",
        consistencyNotes: notes,
        comparableLayerTimeSpreadMs: spread,
      };
    }
  }

  if (notes.length === 0) {
    notes.push(
      "Cadeia temporal coerente para leitura agregada: nenhum downstream com scan bem-sucedido sem upstream, e desvio entre observações recentes dentro do limite operacional.",
    );
  }

  return {
    hasComparableCatalogPocketLayer: c,
    hasComparablePocketEconomicsLayer: e,
    hasComparablePocketExecutionLayer: x,
    hasComparableMinimalPaperLayer: m,
    overallTemporalConsistencyVerdict: "consistent",
    consistencyNotes: notes,
    comparableLayerTimeSpreadMs: spread,
  };
}

export interface ExecutiveSummary {
  headline: string;
  currentStage: string;
  mainConstraint: string;
  nextBestAction: string;
  confidenceNote: string;
}

function firstBlockingFromLayer(layer: Record<string, unknown>): string | null {
  const br = layer.blockingReasons;
  if (!Array.isArray(br) || br.length === 0) return null;
  const s = br[0];
  return typeof s === "string" && s.trim().length > 0 ? s.trim().slice(0, 220) : null;
}

/** Resumo operacional em texto fixo; só usa campos já agregados no digest. */
function buildExecutiveSummary(args: {
  promotionLadder: ProbeSystemLadderDigest["promotionLadder"];
  temporalConsistency: ProbeSystemLadderDigest["temporalConsistency"];
  layers: ProbeSystemLadderDigest["layers"];
}): ExecutiveSummary {
  const pl = args.promotionLadder;
  const tc = args.temporalConsistency;
  const L = args.layers;
  const cat = L.catalogPocket as Record<string, unknown>;
  const econ = L.pocketEconomics as Record<string, unknown>;
  const ex = L.pocketExecution as Record<string, unknown>;
  const min = L.minimalPaperExecution as Record<string, unknown>;

  const tv = tc.overallTemporalConsistencyVerdict;
  const confidenceNote =
    tv === "consistent"
      ? "Coerência temporal: temporally consistent — comparável entre camadas dentro do limite operacional."
      : tv === "partially_stale"
        ? "Coerência temporal: partially stale — desvio temporal entre scans bem-sucedidos; interpretar a escada com cautela."
        : "Coerência temporal: not comparable yet — não interpretar promotionLadder como foto única alinhada no tempo.";

  let currentStage: string;
  if (!tc.hasComparableCatalogPocketLayer) {
    currentStage =
      "Degrau 0 — catalog-pocket sem observação bem-sucedida (base da escada incompleta ou a aguardar primeiro scan).";
  } else if (!tc.hasComparablePocketEconomicsLayer) {
    currentStage = "Degrau 1 — catalog observado; pocket-economics sem scan bem-sucedido ainda.";
  } else if (!tc.hasComparablePocketExecutionLayer) {
    currentStage = "Degrau 2 — economics observada; pocket-execution sem scan bem-sucedido ainda.";
  } else if (!tc.hasComparableMinimalPaperLayer) {
    currentStage = "Degrau 3 — execution observada; minimal paper sem scan bem-sucedido ainda.";
  } else if (pl.executionPromotionVerdict !== "not_ready") {
    currentStage =
      "Degrau 4 — cadeia observada end-to-end; gate de execução permite avanço de paper conforme critérios do probe.";
  } else if (pl.executionObservationVerdict === "minimal_executable_signal") {
    currentStage =
      "Degrau 3+ — observação de execução com sinal mínimo executável; promotion gate de execução ainda not_ready ou borderline.";
  } else {
    currentStage =
      "Degrau 3+ — cadeia observada; foco na promoção de execução (not_ready/borderline) antes de paper novo.";
  }

  let mainConstraint: string;
  if (tv === "not_comparable_yet") {
    mainConstraint =
      tc.consistencyNotes[0] ??
      "Camadas com scans bem-sucedidos desalinhados na cadeia catalog → economics → execution → minimal paper.";
  } else if (tv === "partially_stale") {
    mainConstraint =
      tc.consistencyNotes[0] ??
      "Desvio temporal elevado entre camadas com observação; promotionLadder pode misturar momentos distintos.";
  } else if (pl.executionPromotionVerdict === "not_ready") {
    mainConstraint =
      firstBlockingFromLayer(ex) ??
      "Gate execution promotion em not_ready — novas entradas paper bloqueadas ou limitadas pelo probe mínimo.";
  } else if (pl.economicsPromotionVerdict === "not_ready") {
    mainConstraint =
      firstBlockingFromLayer(econ) ?? "Economics promotion em not_ready — escada ainda não promove execução mínima.";
  } else if (pl.executionObservationVerdict === "blocked") {
    mainConstraint =
      firstBlockingFromLayer(ex) ?? "Observação de execução em blocked — sem sinal executável na última leitura agregada.";
  } else if (!pl.recurrentPocketExists && tc.hasComparableCatalogPocketLayer) {
    mainConstraint =
      "Família alvo sem recurrent_pocket no último catalog observado — substrato recorrente ausente nesta foto.";
  } else {
    mainConstraint = "Nenhum gargalo explícito nas regras resumidas; validar detalhe nos endpoints por camada se necessário.";
  }

  let nextBestAction: string;
  if (tv === "not_comparable_yet") {
    nextBestAction =
      "Concluir primeiro scan bem-sucedido na(s) camada(s) em falta ou alinhar processo/reidratação; ler temporalConsistency.consistencyNotes.";
  } else if (tv === "partially_stale") {
    nextBestAction =
      "Aguardar scans mais recentes nas camadas mais antigas ou rever persistência/higiene se dados estiverem obsoletos.";
  } else if (!tc.hasComparableCatalogPocketLayer) {
    nextBestAction = "Garantir que catalog-pocket corre e completa um scan bem-sucedido antes de conclusões sobre recurrent pocket.";
  } else if (pl.executionPromotionVerdict === "not_ready") {
    nextBestAction =
      "Deixar pocket-execution acumular ciclos; consultar /api/probe/pocket-execution e executionPromotionAssessment.";
  } else if (pl.economicsPromotionVerdict === "not_ready") {
    nextBestAction =
      "Deixar pocket-economics acumular histórico estável; consultar promotionAssessment.blockingReasons no endpoint dedicado.";
  } else if (String(min.primaryVerdict) === "paper_active") {
    nextBestAction =
      "Acompanhar entradas paper abertas até observedWindow no próximo ciclo do minimal-paper-execution.";
  } else {
    nextBestAction =
      "Manter observação agendada; usar endpoints detalhados apenas se precisar de causa raiz ou métricas completas.";
  }

  const catV = String(cat.primaryVerdict ?? "unknown");
  const econV = String(econ.primaryVerdict ?? "unknown");
  const exObs = String(ex.primaryVerdict ?? "unknown");
  const exPr = String(ex.executionPromotionVerdict ?? "unknown");
  const minV = String(min.primaryVerdict ?? "unknown");

  const parts: string[] = [];
  if (!tc.hasComparableCatalogPocketLayer) {
    parts.push("catalog sem scan OK");
  } else {
    parts.push(`catalog ${catV}`);
    if (pl.recurrentPocketExists) parts.push("recurrent_pocket sim");
    else parts.push("recurrent_pocket não");
  }
  parts.push(`economics ${econV}`);
  parts.push(`exec ${exObs}/${exPr}`);
  parts.push(`minimal ${minV}`);
  let headline = `Trilha: ${parts.join("; ")}.`;
  if (headline.length > 200) {
    headline = `${headline.slice(0, 197)}…`;
  }

  return {
    headline,
    currentStage,
    mainConstraint,
    nextBestAction,
    confidenceNote,
  };
}

export interface ProbeSystemLadderDigest {
  computedAt: string;
  summaryVersion: "probe-system-ladder-v1";
  note: string;
  targetFamilyKey: typeof TARGET_FAMILY_KEY;
  layers: {
    catalogPocket: Record<string, unknown>;
    pocketEconomics: Record<string, unknown>;
    pocketExecution: Record<string, unknown>;
    minimalPaperExecution: Record<string, unknown>;
  };
  promotionLadder: {
    recurrentPocketExists: boolean;
    targetFamilyPocketVerdict: string | null;
    economicsPromotionVerdict: string;
    executionObservationVerdict: string;
    executionPromotionVerdict: string;
    minimalPaperExecutionAssessmentVerdict: string;
  };
  temporalConsistency: {
    hasComparableCatalogPocketLayer: boolean;
    hasComparablePocketEconomicsLayer: boolean;
    hasComparablePocketExecutionLayer: boolean;
    hasComparableMinimalPaperLayer: boolean;
    overallTemporalConsistencyVerdict: OverallTemporalConsistencyVerdict;
    consistencyNotes: string[];
    /** ms entre o scan bem-sucedido mais antigo e o mais recente entre camadas comparáveis; null se menos de duas observações */
    comparableLayerTimeSpreadMs: number | null;
    /** Limite operacional usado para partially_stale (ms), documentado para auditoria humana */
    layerTimeSpreadThresholdMs: number;
  };
  executiveSummary: ExecutiveSummary;
  ladderHistory: LadderHistoryDigest;
  ladderTrajectoryAssessment: LadderTrajectoryAssessmentDigest;
  hygiene: {
    paperPersistedStateAnyAppearsTestLike: boolean;
    persistedStateByLayer: Array<{
      layer: string;
      persistedFileExists: boolean;
      statePath: string | null;
      appearsTestLike: boolean;
      hints: string[];
    }>;
  };
  momentumSnipingSummary: {
    scannerRunning: boolean;
    snapshotsTaken: number;
    totalEventsDetected: number;
    capturableEventsCount: number;
    momentumSnipingVerdict: MomentumSnipingDigest["momentumSnipingVerdict"];
    eventFrequencyPerHour: number | null;
    cumulativeConservativeCapturableProxy: number;
    supportingReasons: string[];
    blockingReasons: string[];
    rankedSignalVerdict: RankedSignalVerdict;
    rankedSignalSummaryLine: string;
    topSliceRobustnessVerdict: TopSliceRobustnessVerdict;
    topSliceRobustnessSummaryLine: string;
    selectionAssessmentVerdict: SelectionAssessmentVerdict;
    selectionSummaryLine: string;
    operationalizationVerdict: OperationalizationVerdict;
    operationalizationSummaryLine: string;
    operationalizationV2Verdict: OperationalizationV2Verdict;
    operationalizationV2SummaryLine: string;
    operationalizationRobustnessVerdict: OpsRobustnessVerdict;
    operationalizationRobustnessSummaryLine: string;
    promotionReadinessVerdict: PromotionConfidenceTier;
    promotionReadinessSummaryLine: string;
    promotionProgressSummaryLine: string;
    realisticPaperExecutionVerdict: RealisticPaperVerdict;
    realisticPaperExecutionSummaryLine: string;
    executionSurvivabilityVerdict: ExecutionSurvivabilityVerdict;
    executionSurvivabilitySummaryLine: string;
    segmentedPaperPreparationVerdict: SegmentedPaperPreparationVerdict;
    segmentedPaperPreparationSummaryLine: string;
    segmentedPaperExecutionVerdict: SegmentedPaperExecutionVerdict;
    segmentedPaperExecutionSummaryLine: string;
    segmentedWave2ExecutionVerdict: SegmentedWave2ExecutionVerdict;
    segmentedWave2ExecutionSummaryLine: string;
  };
}

export function buildProbeSystemLadderDigest(): ProbeSystemLadderDigest {
  const t0 = Date.now();
  ensureCatalogPocketProbe();
  ensurePocketEconomicsProbe();
  ensurePocketExecutionProbe();
  ensureMinimalPaperExecutionProbe();
  ensureMomentumSnipingProbe();
  const tAfterEnsure = Date.now();

  const nowMs = Date.now();

  const tBuild0 = Date.now();
  const catalog = buildCatalogPocketDigest();
  const econ = buildPocketEconomicsDigest();
  const exec = buildPocketExecutionDigest();
  const minimal = buildMinimalPaperExecutionDigest({ reusePocketExecutionDigest: exec });
  const hygieneHealth = buildPaperPersistedStateHygieneHealth();
  const tBuild1 = Date.now();

  const targetRow = catalog.familyRows.find(r => r.familyKey === TARGET_FAMILY_KEY);

  const catT = layerTemporalFields(catalog.lastSuccessfulScanAt, nowMs);
  const econT = layerTemporalFields(econ.lastSuccessfulScanAt, nowMs);
  const execT = layerTemporalFields(exec.lastSuccessfulScanAt, nowMs);
  const minT = layerTemporalFields(minimal.lastSuccessfulScanAt, nowMs);

  const temporalConsistency = buildTemporalConsistency({
    nowMs,
    catalog: { lastSuccessfulScanAt: catalog.lastSuccessfulScanAt },
    econ: { lastSuccessfulScanAt: econ.lastSuccessfulScanAt },
    exec: { lastSuccessfulScanAt: exec.lastSuccessfulScanAt },
    minimal: { lastSuccessfulScanAt: minimal.lastSuccessfulScanAt },
  });

  const layers = {
    catalogPocket: {
      layer: "catalog-pocket",
      probeVersion: catalog.probeVersion,
      scanStatus: catalog.scanStatus,
      lastSuccessfulScanAt: catalog.lastSuccessfulScanAt,
      lastMeaningfulObservationAt: catT.lastMeaningfulObservationAt,
      dataFreshness: catT.dataFreshness,
      hasComparableLayer: catT.hasComparableLayer,
      nextScheduledScanAt: catalog.nextScheduledScanAt,
      isScanRunning: catalog.isScanRunning,
      currentRunId: catalog.currentRunId,
      totalScanAttempts: catalog.totalScanAttempts,
      totalScanSuccess: catalog.totalScanSuccess,
      totalScanErrors: catalog.totalScanErrors,
      totalScanSkippedBusy: catalog.totalScanSkippedBusy,
      primaryVerdict: catalog.globalVerdict,
      familiesWithRecurringPocket: catalog.familiesWithRecurringPocket,
      primaryReasons: catalog.lastScanErrorMessage ? [catalog.lastScanErrorMessage] : [],
    },
    pocketEconomics: {
      layer: "pocket-economics",
      probeVersion: econ.probeVersion,
      targetFamilyKey: econ.targetFamilyKey,
      scanStatus: econ.scanStatus,
      lastSuccessfulScanAt: econ.lastSuccessfulScanAt,
      lastMeaningfulObservationAt: econT.lastMeaningfulObservationAt,
      dataFreshness: econT.dataFreshness,
      hasComparableLayer: econT.hasComparableLayer,
      nextScheduledScanAt: econ.nextScheduledScanAt,
      isScanRunning: econ.isScanRunning,
      currentRunId: econ.currentRunId,
      totalScanAttempts: econ.totalScanAttempts,
      totalScanSuccess: econ.totalScanSuccess,
      totalScanErrors: econ.totalScanErrors,
      totalScanSkippedBusy: econ.totalScanSkippedBusy,
      primaryVerdict: econ.promotionAssessment.overallPromotionVerdict,
      supportingReasons: take(econ.promotionAssessment.promotionReasons, 5),
      blockingReasons: take(econ.promotionAssessment.blockingReasons, 5),
      largestStablePocketSize: econ.largestStablePocketSize,
      stableMicroBucketsCount: econ.stableMicroBuckets.length,
    },
    pocketExecution: {
      layer: "pocket-execution",
      probeVersion: exec.probeVersion,
      targetFamilyKey: exec.targetFamilyKey,
      scanStatus: exec.scanStatus,
      lastSuccessfulScanAt: exec.lastSuccessfulScanAt,
      lastMeaningfulObservationAt: execT.lastMeaningfulObservationAt,
      dataFreshness: execT.dataFreshness,
      hasComparableLayer: execT.hasComparableLayer,
      nextScheduledScanAt: exec.nextScheduledScanAt,
      isScanRunning: exec.isScanRunning,
      currentRunId: exec.currentRunId,
      totalScanAttempts: exec.totalScanAttempts,
      totalScanSuccess: exec.totalScanSuccess,
      totalScanErrors: exec.totalScanErrors,
      totalScanSkippedBusy: exec.totalScanSkippedBusy,
      primaryVerdict: exec.executionObservationVerdict,
      executionPromotionVerdict: exec.executionPromotionAssessment.overallExecutionPromotionVerdict,
      supportingReasons: take(exec.supportingReasons, 5),
      blockingReasons: take(exec.blockingReasons, 5),
      stableMicroKeysAtScanCount: exec.stableMicroKeysAtScan.length,
      candidateExecutionPocketsCount: exec.candidateExecutionPockets.length,
    },
    minimalPaperExecution: {
      layer: "minimal-paper-execution",
      probeVersion: minimal.probeVersion,
      targetFamilyKey: minimal.targetFamilyKey,
      scanStatus: minimal.scanStatus,
      lastSuccessfulScanAt: minimal.lastSuccessfulScanAt,
      lastMeaningfulObservationAt: minT.lastMeaningfulObservationAt,
      dataFreshness: minT.dataFreshness,
      hasComparableLayer: minT.hasComparableLayer,
      nextScheduledScanAt: minimal.nextScheduledScanAt,
      isScanRunning: minimal.isScanRunning,
      currentRunId: minimal.currentRunId,
      totalScanAttempts: minimal.totalScanAttempts,
      totalScanSuccess: minimal.totalScanSuccess,
      totalScanErrors: minimal.totalScanErrors,
      totalScanSkippedBusy: minimal.totalScanSkippedBusy,
      primaryVerdict: minimal.minimalPaperExecutionAssessment.assessmentVerdict,
      gateOverallExecutionPromotionVerdict:
        minimal.minimalPaperExecutionAssessment.gateOverallExecutionPromotionVerdict,
      supportingReasons: take(minimal.minimalPaperExecutionAssessment.supportingReasons, 5),
      blockingReasons: take(minimal.minimalPaperExecutionAssessment.blockingReasons, 5),
      paperEntriesCount: minimal.entries.length,
      openPaperEntriesCount: minimal.minimalPaperExecutionAssessment.openPaperEntriesCount,
      observedPaperEntriesCount: minimal.minimalPaperExecutionAssessment.observedEntriesCount,
      microEdgeReadVerdict: minimal.microEdgeAssessment.microEdgeReadVerdict,
      microEdgeEnoughSample: minimal.microEdgeAssessment.enoughSampleForMicroEdgeRead,
      microEdgeEligibleClosed: minimal.microEdgeAssessment.eligibleClosedEpisodesForMicroRead,
      microEdgePositiveRate: minimal.microEdgeAssessment.microPositiveRate,
      microEdgeNegativeRate: minimal.microEdgeAssessment.microNegativeRate,
      microEdgeSummaryLine: buildMicroEdgeSummaryLine(minimal.microEdgeAssessment),
      microEdgeSupportingReasons: take(minimal.microEdgeAssessment.supportingReasons, 3),
      microEdgeBlockingReasons: take(minimal.microEdgeAssessment.blockingReasons, 3),
    },
  };

  const persistedStateByLayer = [
    {
      layer: "catalog-pocket",
      persistedFileExists: false,
      statePath: null as string | null,
      appearsTestLike: false,
      hints: [] as string[],
    },
    ...hygieneHealth.paperPersistedStateHygieneProbes.map(r => ({
      layer: r.probe,
      persistedFileExists: r.fileExists,
      statePath: r.statePath,
      appearsTestLike: r.appearsTestLike,
      hints: r.hints,
    })),
  ];

  const promotionLadder = {
    recurrentPocketExists: targetRow?.familyPocketVerdict === "recurrent_pocket",
    targetFamilyPocketVerdict: targetRow?.familyPocketVerdict ?? null,
    economicsPromotionVerdict: econ.promotionAssessment.overallPromotionVerdict,
    executionObservationVerdict: exec.executionObservationVerdict,
    executionPromotionVerdict: exec.executionPromotionAssessment.overallExecutionPromotionVerdict,
    minimalPaperExecutionAssessmentVerdict: minimal.minimalPaperExecutionAssessment.assessmentVerdict,
  };

  const temporalConsistencyBlock = {
    ...temporalConsistency,
    layerTimeSpreadThresholdMs: TEMPORAL_SPREAD_STALE_MS,
  };

  const executiveSummary = buildExecutiveSummary({
    promotionLadder,
    temporalConsistency: temporalConsistencyBlock,
    layers,
  });

  const tHist0 = Date.now();
  const { ladderHistory, ladderTrajectoryAssessment } = processLadderHistoryForDigest({
    promotionLadder,
    temporalConsistencyVerdict: temporalConsistencyBlock.overallTemporalConsistencyVerdict,
    currentStage: executiveSummary.currentStage,
  });
  const tEnd = Date.now();
  console.log(
    `[system-ladder] totalMs=${tEnd - t0} ensuresMs=${tAfterEnsure - t0} digestBlocksMs=${tBuild1 - tBuild0} aggregateMs=${tHist0 - tBuild1} historyMs=${tEnd - tHist0}`,
  );

  const msd = buildMomentumSnipingDigest();
  const totalMsEvents = msd.candidateMomentumEventsCount + msd.candidateSnipingEventsCount;
  const capturableMsEvents = msd.recentEvents.filter(e => e.capturable).length;
  const momentumSnipingSummary = {
    scannerRunning: msd.scannerRunning,
    snapshotsTaken: msd.snapshotsTaken,
    totalEventsDetected: totalMsEvents,
    capturableEventsCount: capturableMsEvents,
    momentumSnipingVerdict: msd.momentumSnipingVerdict,
    eventFrequencyPerHour: msd.eventFrequencyPerHour,
    cumulativeConservativeCapturableProxy: msd.cumulativeConservativeCapturableProxy,
    supportingReasons: take(msd.supportingReasons, 3),
    blockingReasons: take(msd.blockingReasons, 3),
    rankedSignalVerdict: msd.rankedEventAssessment.rankedSignalVerdict,
    rankedSignalSummaryLine: buildRankedSignalSummaryLine(msd.rankedEventAssessment),
    topSliceRobustnessVerdict: msd.topSliceRobustnessAssessment.topSliceRobustnessVerdict,
    topSliceRobustnessSummaryLine: buildRobustnessSummaryLine(msd.topSliceRobustnessAssessment),
    selectionAssessmentVerdict: msd.topSliceSelectionAssessment.selectionAssessmentVerdict,
    selectionSummaryLine: buildSelectionSummaryLine(msd.topSliceSelectionAssessment),
    operationalizationVerdict: msd.operationalizationAssessment.operationalizationVerdict,
    operationalizationSummaryLine: buildOperationalizationSummaryLine(msd.operationalizationAssessment),
    operationalizationV2Verdict: msd.operationalizationAssessmentV2.operationalizationVerdict,
    operationalizationV2SummaryLine: buildOperationalizationV2SummaryLine(msd.operationalizationAssessmentV2),
    operationalizationRobustnessVerdict: msd.operationalizationRobustnessAssessment.robustnessVerdict,
    operationalizationRobustnessSummaryLine: buildOpsRobustnessSummaryLine(msd.operationalizationRobustnessAssessment),
    promotionReadinessVerdict: msd.promotionReadinessAssessment.promotionReadinessVerdict,
    promotionReadinessSummaryLine: buildPromotionReadinessSummaryLine(msd.promotionReadinessAssessment),
    promotionProgressSummaryLine: buildProgressSummaryLine(msd.promotionProgressAssessment),
    realisticPaperExecutionVerdict: msd.realisticPaperExecutionAssessment.realisticPaperExecutionVerdict,
    realisticPaperExecutionSummaryLine: buildRealisticPaperSummaryLine(msd.realisticPaperExecutionAssessment),
    executionSurvivabilityVerdict: msd.executionSurvivabilitySegmentation.executionSurvivabilityVerdict,
    executionSurvivabilitySummaryLine:
      msd.executionSurvivabilitySegmentation.executionSurvivabilitySummaryLine,
    segmentedPaperPreparationVerdict:
      msd.segmentedPaperTestPreparation.segmentedPaperPreparationVerdict,
    segmentedPaperPreparationSummaryLine:
      msd.segmentedPaperTestPreparation.segmentedPaperPreparationSummaryLine,
    segmentedPaperExecutionVerdict:
      msd.segmentedPaperExecutionAssessment.segmentedPaperExecutionVerdict,
    segmentedPaperExecutionSummaryLine:
      msd.segmentedPaperExecutionAssessment.segmentedPaperExecutionSummaryLine,
    segmentedWave2ExecutionVerdict:
      msd.segmentedPaperExecutionWave2Assessment.segmentedWave2ExecutionVerdict,
    segmentedWave2ExecutionSummaryLine:
      msd.segmentedPaperExecutionWave2Assessment.segmentedWave2ExecutionSummaryLine,
  };

  return {
    computedAt: new Date().toISOString(),
    summaryVersion: "probe-system-ladder-v1",
    note:
      "Agregação leve para leitura rápida. Critérios e payloads completos permanecem nos endpoints /api/probe/catalog-pocket, pocket-economics, pocket-execution, minimal-paper-execution. Histórico agregado (ladderHistory): system-ladder-history.json sob PAPER_STATE_DIR ou cwd/.paper; SYSTEM_LADDER_HISTORY_PATH; SYSTEM_LADDER_HISTORY_DISABLE_DISK=1; SYSTEM_LADDER_HISTORY_MAX_SNAPSHOTS (5–50, default 15).",
    targetFamilyKey: TARGET_FAMILY_KEY,
    layers,
    promotionLadder,
    temporalConsistency: temporalConsistencyBlock,
    executiveSummary,
    ladderHistory,
    ladderTrajectoryAssessment,
    hygiene: {
      paperPersistedStateAnyAppearsTestLike: hygieneHealth.paperPersistedStateAnyAppearsTestLike,
      persistedStateByLayer,
    },
    momentumSnipingSummary,
  };
}
