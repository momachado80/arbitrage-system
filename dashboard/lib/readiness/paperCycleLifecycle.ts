/**
 * Microcapital Readiness Gate — paper cycle lifecycle (paper-only).
 *
 * Pure functions that turn paper execution assessments + markout follow-ups
 * into observable lifecycle events:
 *
 *   paper_cycle_opened — derived from a `paper_cycle_positive` assessment
 *   paper_cycle_closed — only emitted when there is markout evidence
 *   paper_cycle_insufficient_evidence — fallback when markouts are absent
 *
 * GUARDRAILS — this module:
 *   - is pure / read-only (no I/O)
 *   - does NOT submit any order
 *   - does NOT call any financial endpoint
 *   - does NOT sign any transaction
 *   - does NOT touch a wallet, signer, or private key
 *   - does NOT authorize real execution or capital activation
 *   - NEVER converts an assessment into a closed trade automatically
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import type { PaperExecutionAssessment } from "./paperExecutionAssessmentParser";
import type {
  MarkoutFollowupRecord,
  MarkoutHorizonMs,
} from "./markoutFollowupProducer";

export type PaperCycleCloseStatus =
  | "closed"
  | "insufficient_evidence";

export type PaperCycleCloseReason =
  | "markout_window_complete"
  | "markout_insufficient";

export type PaperCycleEvidenceSource =
  | "markout_followups"
  | "no_followups";

export interface PaperCycleOpenedEvent {
  type: "paper_cycle_opened";
  cycleId: string;
  marketId: string | null;
  openedAt: string;
  estimatedEntryNet: number | null;
}

export interface PaperCycleClosedEvent {
  type: "paper_cycle_closed";
  cycleId: string;
  marketId: string | null;
  openedAt: string;
  closedAt: string;
  closeStatus: "closed";
  closeReason: PaperCycleCloseReason;
  evidenceSource: PaperCycleEvidenceSource;
  estimatedEntryNet: number | null;
  estimatedExitNet: number | null;
  /**
   * Simulated paper-only PnL = estimatedEntryNet + averageMarkout.
   * NOT realized PnL. NOT capital. NOT a trade fill.
   */
  simulatedNetPnl: number;
  averageMarkout: number;
  markoutHorizonsObserved: MarkoutHorizonMs[];
}

export interface PaperCycleInsufficientEvidenceEvent {
  type: "paper_cycle_insufficient_evidence";
  cycleId: string;
  marketId: string | null;
  openedAt: string;
  closeStatus: "insufficient_evidence";
  evidenceSource: PaperCycleEvidenceSource;
  reason: "markout_followups_missing" | "markout_partial";
  estimatedEntryNet: number | null;
  observedHorizons: MarkoutHorizonMs[];
}

export type PaperCycleEvent =
  | PaperCycleOpenedEvent
  | PaperCycleClosedEvent
  | PaperCycleInsufficientEvidenceEvent;

export interface PaperCycleLifecycleSummary {
  openedPaperCycles: number;
  closedPaperCycles: number;
  insufficientEvidencePaperCycles: number;
  simulatedClosedCycleNetTotal: number;
  simulatedClosedCyclePositiveCount: number;
  simulatedClosedCycleNegativeCount: number;
  markoutBackedCycleCount: number;
  events: PaperCycleEvent[];
}

const REQUIRED_HORIZONS: MarkoutHorizonMs[] = [5_000, 30_000, 60_000];

/**
 * Build a deterministic cycle ID from an assessment.
 *
 * Format: `pc::<marketId>::<isoTimestamp>`. If marketId or isoTimestamp is
 * missing, fall back to "unknown".
 */
export function buildPaperCycleId(assessment: PaperExecutionAssessment): string {
  const market = assessment.marketId ?? "unknown";
  const ts = assessment.isoTimestamp ?? "unknown";
  return `pc::${market}::${ts}`;
}

function isPositiveAssessment(a: PaperExecutionAssessment): boolean {
  return (a.paperExecutionVerdict ?? "").toLowerCase() === "paper_cycle_positive";
}

function followupsForCycle(
  cycleId: string,
  followups: ReadonlyArray<MarkoutFollowupRecord>,
): MarkoutFollowupRecord[] {
  return followups.filter(f => f.cycleId === cycleId);
}

function averageMarkout(records: ReadonlyArray<MarkoutFollowupRecord>): number {
  if (records.length === 0) return 0;
  const sum = records.reduce((s, r) => s + r.markout, 0);
  return sum / records.length;
}

/**
 * Convert a positive assessment into a `paper_cycle_opened` event.
 *
 * NEVER returns a closed event automatically — closure is only computed when
 * markouts are joined via {@link buildPaperCycleEvents}.
 */
export function assessmentToOpenedCycle(
  assessment: PaperExecutionAssessment,
): PaperCycleOpenedEvent | null {
  if (!isPositiveAssessment(assessment)) return null;
  if (!assessment.isoTimestamp) return null;
  return {
    type: "paper_cycle_opened",
    cycleId: buildPaperCycleId(assessment),
    marketId: assessment.marketId,
    openedAt: assessment.isoTimestamp,
    estimatedEntryNet: assessment.estimatedNetAfterPaperExecution,
  };
}

/**
 * Build the full lifecycle event stream for a list of assessments + followups.
 *
 * Closure rule: a cycle is closed only when ALL three required horizons
 * (5s/30s/60s) are present in the followups for that cycle. Otherwise we emit
 * an `insufficient_evidence` event.
 */
export function buildPaperCycleEvents(
  assessments: ReadonlyArray<PaperExecutionAssessment>,
  followups: ReadonlyArray<MarkoutFollowupRecord>,
): PaperCycleEvent[] {
  const events: PaperCycleEvent[] = [];
  for (const a of assessments) {
    const opened = assessmentToOpenedCycle(a);
    if (!opened) continue;
    events.push(opened);

    const cycleId = opened.cycleId;
    const cf = followupsForCycle(cycleId, followups);
    const horizonsObserved = cf
      .map(f => f.horizonMs)
      .filter((h): h is MarkoutHorizonMs =>
        REQUIRED_HORIZONS.includes(h as MarkoutHorizonMs),
      );
    const allRequired = REQUIRED_HORIZONS.every(h => horizonsObserved.includes(h));

    if (cf.length === 0) {
      events.push({
        type: "paper_cycle_insufficient_evidence",
        cycleId,
        marketId: opened.marketId,
        openedAt: opened.openedAt,
        closeStatus: "insufficient_evidence",
        evidenceSource: "no_followups",
        reason: "markout_followups_missing",
        estimatedEntryNet: opened.estimatedEntryNet,
        observedHorizons: [],
      });
      continue;
    }

    if (!allRequired) {
      events.push({
        type: "paper_cycle_insufficient_evidence",
        cycleId,
        marketId: opened.marketId,
        openedAt: opened.openedAt,
        closeStatus: "insufficient_evidence",
        evidenceSource: "markout_followups",
        reason: "markout_partial",
        estimatedEntryNet: opened.estimatedEntryNet,
        observedHorizons: Array.from(new Set(horizonsObserved)),
      });
      continue;
    }

    const avgMarkout = averageMarkout(cf);
    const entryNet = opened.estimatedEntryNet ?? 0;
    const simulatedNetPnl = entryNet + avgMarkout;
    const closedAt = cf
      .map(r => r.followupObservedAt)
      .reduce((mx, n) => (n > mx ? n : mx), 0);

    events.push({
      type: "paper_cycle_closed",
      cycleId,
      marketId: opened.marketId,
      openedAt: opened.openedAt,
      closedAt: new Date(closedAt).toISOString(),
      closeStatus: "closed",
      closeReason: "markout_window_complete",
      evidenceSource: "markout_followups",
      estimatedEntryNet: opened.estimatedEntryNet,
      estimatedExitNet: avgMarkout,
      simulatedNetPnl,
      averageMarkout: avgMarkout,
      markoutHorizonsObserved: Array.from(new Set(horizonsObserved)),
    });
  }
  return events;
}

/**
 * Aggregate lifecycle events into a summary safe to publish in the dossier.
 *
 * NEVER counts assessments as closed cycles. NEVER infers realized PnL — the
 * "simulatedNetPnl" total is explicitly labelled as paper-only.
 */
export function summarizePaperCycleLifecycle(
  assessments: ReadonlyArray<PaperExecutionAssessment>,
  followups: ReadonlyArray<MarkoutFollowupRecord>,
): PaperCycleLifecycleSummary {
  const events = buildPaperCycleEvents(assessments, followups);
  let opened = 0;
  let closed = 0;
  let insufficient = 0;
  let netTotal = 0;
  let positives = 0;
  let negatives = 0;
  let markoutBacked = 0;

  for (const e of events) {
    if (e.type === "paper_cycle_opened") opened++;
    else if (e.type === "paper_cycle_closed") {
      closed++;
      markoutBacked++;
      netTotal += e.simulatedNetPnl;
      if (e.simulatedNetPnl > 0) positives++;
      else if (e.simulatedNetPnl < 0) negatives++;
    } else if (e.type === "paper_cycle_insufficient_evidence") {
      insufficient++;
    }
  }

  return {
    openedPaperCycles: opened,
    closedPaperCycles: closed,
    insufficientEvidencePaperCycles: insufficient,
    simulatedClosedCycleNetTotal: netTotal,
    simulatedClosedCyclePositiveCount: positives,
    simulatedClosedCycleNegativeCount: negatives,
    markoutBackedCycleCount: markoutBacked,
    events,
  };
}

export const EMPTY_PAPER_CYCLE_LIFECYCLE_SUMMARY: PaperCycleLifecycleSummary = {
  openedPaperCycles: 0,
  closedPaperCycles: 0,
  insufficientEvidencePaperCycles: 0,
  simulatedClosedCycleNetTotal: 0,
  simulatedClosedCyclePositiveCount: 0,
  simulatedClosedCycleNegativeCount: 0,
  markoutBackedCycleCount: 0,
  events: [],
};
