/**
 * Microcapital Readiness Gate — paper execution assessment parser.
 *
 * Reads `cross-venue-anchor-1823789-paper-execution-history.jsonl` (paper-only
 * audit history written by the cross-venue-anchor 1823789 paper-execution
 * worker) and extracts per-line *assessment* records. Assessments are NOT
 * closed paper trades — they describe whether each cycle would have been
 * worth executing on paper, with simulated entry/exit slippage, hedge cost,
 * etc. The gate must NOT count assessments as realized PnL or closed cycles.
 *
 * GUARDRAILS — this module:
 *   - is read-only (filesystem read of an audit JSONL file)
 *   - does NOT submit any order
 *   - does NOT call any financial endpoint
 *   - does NOT sign any transaction
 *   - does NOT touch a wallet, signer, or private key
 *   - does NOT authorize real execution or capital activation
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";

export interface PaperExecutionAssessment {
  isoTimestamp: string | null;
  marketId: string | null;
  marketTitle: string | null;
  paperExecutionVerdict: string | null;
  executionGateVerdict: string | null;
  narrowValidationVerdict: string | null;
  estimatedNetBeforeExecution: number | null;
  estimatedNetAfterPaperExecution: number | null;
  estimatedEntrySlippage: number | null;
  estimatedExitSlippage: number | null;
  estimatedHedgeCost: number | null;
  estimatedEntryCost: number | null;
  estimatedExitCost: number | null;
  estimatedLegRiskCost: number | null;
  microstructureObservationState: string | null;
  microstructureDetail: string | null;
  clobBookStructure: string | null;
  clobBestBid: number | null;
  clobBestAsk: number | null;
  clobMid: number | null;
  clobSpreadUsed: number | null;
  clobBidLevels: number | null;
  clobAskLevels: number | null;
  entryDepthAvailable: boolean | null;
  exitDepthAvailable: boolean | null;
  supportingNote: string | null;
}

export interface PaperExecutionAssessmentSummary {
  paperExecutionAssessmentCount: number;
  paperCyclePositiveCount: number;
  paperBlockedByGateCount: number;
  paperNotViableUnderStressCount: number;
  paperBlockedByExecutionCount: number;
  clobUnavailableCount: number;
  estimatedNetAfterPaperExecutionTotal: number;
  estimatedNetAfterPaperExecutionPositiveTotal: number;
  estimatedNetAfterPaperExecutionMedian: number;
  estimatedNetAfterPaperExecutionBest: number;
  estimatedNetAfterPaperExecutionWorst: number;
  positiveAssessmentRate: number;
  blockedByGateRate: number;
  firstAssessmentAt: string | null;
  lastAssessmentAt: string | null;
}

function asString(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}
function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function asBoolean(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

/**
 * Parse a single JSONL line into a {@link PaperExecutionAssessment}.
 * Returns `null` for blank lines or lines that cannot be JSON-parsed.
 *
 * Field extraction is tolerant: unknown keys are ignored; missing keys yield
 * null. The caller decides whether absence-of-data should affect verdicts.
 */
export function parsePaperExecutionAssessmentLine(
  line: string,
): PaperExecutionAssessment | null {
  const trimmed = line.trim();
  if (trimmed.length === 0) return null;
  let obj: Record<string, unknown>;
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    obj = parsed as Record<string, unknown>;
  } catch {
    return null;
  }

  // Some rows nest the gate digest under a key (e.g. "gate"); flatten lookups.
  const gate = (obj.gate as Record<string, unknown> | undefined) ?? {};
  const supporting = (obj.supportingNote as Record<string, unknown> | string | undefined);

  function pick(key: string): unknown {
    if (key in obj) return obj[key];
    if (key in gate) return gate[key];
    return undefined;
  }

  return {
    isoTimestamp:
      asString(obj.isoTimestamp) ??
      asString(obj.timestamp) ??
      asString(obj.observedAt) ??
      null,
    marketId: asString(pick("marketId")),
    marketTitle: asString(pick("marketTitle")),
    paperExecutionVerdict: asString(pick("paperExecutionVerdict")),
    executionGateVerdict: asString(pick("executionGateVerdict")),
    narrowValidationVerdict: asString(pick("narrowValidationVerdict")),
    estimatedNetBeforeExecution: asNumber(pick("estimatedNetBeforeExecution")),
    estimatedNetAfterPaperExecution: asNumber(pick("estimatedNetAfterPaperExecution")),
    estimatedEntrySlippage: asNumber(pick("estimatedEntrySlippage")),
    estimatedExitSlippage: asNumber(pick("estimatedExitSlippage")),
    estimatedHedgeCost: asNumber(pick("estimatedHedgeCost")),
    estimatedEntryCost: asNumber(pick("estimatedEntryCost")),
    estimatedExitCost: asNumber(pick("estimatedExitCost")),
    estimatedLegRiskCost: asNumber(pick("estimatedLegRiskCost")),
    microstructureObservationState: asString(pick("microstructureObservationState")),
    microstructureDetail: asString(pick("microstructureDetail")),
    clobBookStructure: asString(pick("clobBookStructure")),
    clobBestBid: asNumber(pick("clobBestBid")),
    clobBestAsk: asNumber(pick("clobBestAsk")),
    clobMid: asNumber(pick("clobMid")),
    clobSpreadUsed: asNumber(pick("clobSpreadUsed")),
    clobBidLevels: asNumber(pick("clobBidLevels")),
    clobAskLevels: asNumber(pick("clobAskLevels")),
    entryDepthAvailable: asBoolean(pick("entryDepthAvailable")),
    exitDepthAvailable: asBoolean(pick("exitDepthAvailable")),
    supportingNote:
      typeof supporting === "string"
        ? supporting
        : asString(pick("supportingNote")) ??
          (supporting && typeof supporting === "object"
            ? JSON.stringify(supporting)
            : null),
  };
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

/**
 * Aggregate a list of assessments into a summary. NEVER converts assessments
 * into closed paper cycles. NEVER infers realized PnL. The summary is
 * statistical observation only.
 */
export function summarizePaperExecutionAssessments(
  assessments: ReadonlyArray<PaperExecutionAssessment>,
): PaperExecutionAssessmentSummary {
  const total = assessments.length;
  let positive = 0;
  let blockedByGate = 0;
  let notViable = 0;
  let blockedByExecution = 0;
  let clobUnavailable = 0;
  let netTotal = 0;
  let netPositiveTotal = 0;
  const netValues: number[] = [];
  let firstAt: string | null = null;
  let lastAt: string | null = null;

  for (const a of assessments) {
    const verdict = (a.paperExecutionVerdict ?? "").toLowerCase();
    const gateVerdict = (a.executionGateVerdict ?? "").toLowerCase();
    const microState = (a.microstructureObservationState ?? "").toLowerCase();

    if (verdict === "paper_cycle_positive") positive++;
    if (verdict === "blocked_by_gate") blockedByGate++;
    if (verdict === "not_viable_under_stress") notViable++;
    if (gateVerdict === "blocked_by_execution") blockedByExecution++;
    if (microState === "clob_book_unavailable") clobUnavailable++;

    const net = a.estimatedNetAfterPaperExecution;
    if (typeof net === "number" && Number.isFinite(net)) {
      netTotal += net;
      if (net > 0) netPositiveTotal += net;
      netValues.push(net);
    }

    if (a.isoTimestamp) {
      if (firstAt === null || a.isoTimestamp < firstAt) firstAt = a.isoTimestamp;
      if (lastAt === null || a.isoTimestamp > lastAt) lastAt = a.isoTimestamp;
    }
  }

  const positiveAssessmentRate = total > 0 ? positive / total : 0;
  const blockedByGateRate = total > 0 ? blockedByGate / total : 0;

  const best = netValues.length > 0 ? Math.max(...netValues) : 0;
  const worst = netValues.length > 0 ? Math.min(...netValues) : 0;

  return {
    paperExecutionAssessmentCount: total,
    paperCyclePositiveCount: positive,
    paperBlockedByGateCount: blockedByGate,
    paperNotViableUnderStressCount: notViable,
    paperBlockedByExecutionCount: blockedByExecution,
    clobUnavailableCount: clobUnavailable,
    estimatedNetAfterPaperExecutionTotal: netTotal,
    estimatedNetAfterPaperExecutionPositiveTotal: netPositiveTotal,
    estimatedNetAfterPaperExecutionMedian: median(netValues),
    estimatedNetAfterPaperExecutionBest: best,
    estimatedNetAfterPaperExecutionWorst: worst,
    positiveAssessmentRate,
    blockedByGateRate,
    firstAssessmentAt: firstAt,
    lastAssessmentAt: lastAt,
  };
}

/**
 * Read a JSONL file containing paper execution assessments. Returns an empty
 * list if the file does not exist. Lines that cannot be parsed are skipped.
 */
export function readPaperExecutionAssessmentsFromJsonl(
  filePath: string,
): PaperExecutionAssessment[] {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return [];
  }
  const out: PaperExecutionAssessment[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const r = parsePaperExecutionAssessmentLine(line);
    if (r !== null) out.push(r);
  }
  return out;
}

export const EMPTY_PAPER_EXECUTION_ASSESSMENT_SUMMARY: PaperExecutionAssessmentSummary = {
  paperExecutionAssessmentCount: 0,
  paperCyclePositiveCount: 0,
  paperBlockedByGateCount: 0,
  paperNotViableUnderStressCount: 0,
  paperBlockedByExecutionCount: 0,
  clobUnavailableCount: 0,
  estimatedNetAfterPaperExecutionTotal: 0,
  estimatedNetAfterPaperExecutionPositiveTotal: 0,
  estimatedNetAfterPaperExecutionMedian: 0,
  estimatedNetAfterPaperExecutionBest: 0,
  estimatedNetAfterPaperExecutionWorst: 0,
  positiveAssessmentRate: 0,
  blockedByGateRate: 0,
  firstAssessmentAt: null,
  lastAssessmentAt: null,
};
