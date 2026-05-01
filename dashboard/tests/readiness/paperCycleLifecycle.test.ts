/**
 * Tests — paper cycle lifecycle (paper-only).
 *
 * Paper/shadow only. NEVER converts an assessment into a closed trade.
 * NEVER infers realized PnL. NEVER touches submit/wallet/signer/private key.
 * Policy: shadow_only_no_api_submit_v1.
 */

import {
  assessmentToOpenedCycle,
  buildPaperCycleEvents,
  buildPaperCycleId,
  summarizePaperCycleLifecycle,
} from "../../lib/readiness/paperCycleLifecycle";
import type { PaperExecutionAssessment } from "../../lib/readiness/paperExecutionAssessmentParser";
import type { MarkoutFollowupRecord } from "../../lib/readiness/markoutFollowupProducer";

import { assertEqual, assertTrue, describe, test } from "./_assert";

const ISO_BASE = "2026-04-29T12:00:00.000Z";
const TS_BASE = Date.parse(ISO_BASE);

function positiveAssessment(overrides: Partial<PaperExecutionAssessment> = {}): PaperExecutionAssessment {
  return {
    isoTimestamp: ISO_BASE,
    marketId: "1823789",
    marketTitle: "Cross-venue anchor 1823789",
    paperExecutionVerdict: "paper_cycle_positive",
    executionGateVerdict: "passed",
    narrowValidationVerdict: "passed",
    estimatedNetBeforeExecution: 0.02,
    estimatedNetAfterPaperExecution: 0.015,
    estimatedEntrySlippage: null,
    estimatedExitSlippage: null,
    estimatedHedgeCost: null,
    estimatedEntryCost: null,
    estimatedExitCost: null,
    estimatedLegRiskCost: null,
    microstructureObservationState: "clob_book_observed",
    microstructureDetail: null,
    clobBookStructure: "two_sided",
    clobBestBid: 0.49,
    clobBestAsk: 0.51,
    clobMid: 0.5,
    clobSpreadUsed: 0.02,
    clobBidLevels: 5,
    clobAskLevels: 5,
    entryDepthAvailable: true,
    exitDepthAvailable: true,
    supportingNote: null,
    ...overrides,
  };
}

function blockedAssessment(): PaperExecutionAssessment {
  return positiveAssessment({ paperExecutionVerdict: "blocked_by_gate" });
}

function followup(
  cycleId: string,
  horizonMs: 5_000 | 30_000 | 60_000,
  followupPrice: number,
): MarkoutFollowupRecord {
  return {
    type: "markout_followup",
    cycleId,
    marketId: "1823789",
    baseObservedAt: TS_BASE,
    followupObservedAt: TS_BASE + horizonMs,
    horizonMs,
    basePrice: 0.5,
    followupPrice,
    markout: followupPrice - 0.5,
    source: "paper_shadow_readonly",
  };
}

describe("tests/readiness/paperCycleLifecycle.test.ts", () => {
  test("positive assessment becomes opened, NOT closed automatically", () => {
    const a = positiveAssessment();
    const opened = assessmentToOpenedCycle(a);
    assertTrue(opened !== null, "opened event produced");
    assertEqual(opened!.type, "paper_cycle_opened", "type=opened");
    assertEqual(opened!.cycleId, `pc::1823789::${ISO_BASE}`, "cycleId deterministic");
  });

  test("non-positive assessment does NOT produce opened event", () => {
    const opened = assessmentToOpenedCycle(blockedAssessment());
    assertEqual(opened, null, "no opened event for blocked assessment");
  });

  test("opened cycle without followups → insufficient_evidence", () => {
    const events = buildPaperCycleEvents([positiveAssessment()], []);
    assertEqual(events.length, 2, "opened + insufficient_evidence");
    assertEqual(events[0].type, "paper_cycle_opened", "first is opened");
    assertEqual(
      events[1].type,
      "paper_cycle_insufficient_evidence",
      "second is insufficient_evidence",
    );
  });

  test("opened cycle with PARTIAL markouts → insufficient_evidence (markout_partial)", () => {
    const a = positiveAssessment();
    const cycleId = buildPaperCycleId(a);
    const events = buildPaperCycleEvents(
      [a],
      [followup(cycleId, 5_000, 0.51)],
    );
    const last = events[events.length - 1];
    assertEqual(last.type, "paper_cycle_insufficient_evidence", "still insufficient");
    if (last.type === "paper_cycle_insufficient_evidence") {
      assertEqual(last.reason, "markout_partial", "partial reason");
    }
  });

  test("opened cycle with ALL three horizons present → closed; simulatedNetPnl computed", () => {
    const a = positiveAssessment({ estimatedNetAfterPaperExecution: 0.01 });
    const cycleId = buildPaperCycleId(a);
    const events = buildPaperCycleEvents(
      [a],
      [
        followup(cycleId, 5_000, 0.505),  // markout +0.005
        followup(cycleId, 30_000, 0.51),  // markout +0.010
        followup(cycleId, 60_000, 0.515), // markout +0.015
      ],
    );
    const closed = events.find(e => e.type === "paper_cycle_closed");
    assertTrue(closed !== undefined, "closed event present");
    if (closed && closed.type === "paper_cycle_closed") {
      assertEqual(closed.closeStatus, "closed", "closeStatus=closed");
      assertEqual(closed.closeReason, "markout_window_complete", "complete window");
      // entryNet=0.01, avgMarkout=(0.005+0.010+0.015)/3=0.010 → simulatedNetPnl=0.020
      assertTrue(
        Math.abs(closed.simulatedNetPnl - 0.02) < 1e-9,
        `simulatedNetPnl ≈ 0.02 (got ${closed.simulatedNetPnl})`,
      );
    }
  });

  test("cycleId is deterministic for the same assessment", () => {
    const a = positiveAssessment();
    assertEqual(buildPaperCycleId(a), buildPaperCycleId(a), "deterministic");
    assertEqual(buildPaperCycleId(a), `pc::1823789::${ISO_BASE}`, "format");
  });

  test("summary aggregates correctly", () => {
    const a1 = positiveAssessment({ isoTimestamp: ISO_BASE });
    const a2 = positiveAssessment({
      isoTimestamp: "2026-04-29T13:00:00.000Z",
      estimatedNetAfterPaperExecution: -0.005,
    });
    const blocked = blockedAssessment();
    const c1 = buildPaperCycleId(a1);
    const c2 = buildPaperCycleId(a2);
    const s = summarizePaperCycleLifecycle(
      [a1, a2, blocked],
      [
        followup(c1, 5_000, 0.51),
        followup(c1, 30_000, 0.512),
        followup(c1, 60_000, 0.515),
        // c2 has no followups → insufficient evidence
      ],
    );
    assertEqual(s.openedPaperCycles, 2, "2 opened (both positives)");
    assertEqual(s.closedPaperCycles, 1, "1 closed (only c1 has full markouts)");
    assertEqual(s.insufficientEvidencePaperCycles, 1, "1 insufficient (c2)");
    assertEqual(s.markoutBackedCycleCount, 1, "1 markout-backed");
    assertTrue(s.simulatedClosedCycleNetTotal > 0, "positive simulated total");
    assertEqual(s.simulatedClosedCyclePositiveCount, 1, "1 positive close");
  });

  test("paperCycleLifecycle module surface contains no submit/wallet/signer/privateKey", () => {
    const surface = [
      "buildPaperCycleId",
      "assessmentToOpenedCycle",
      "buildPaperCycleEvents",
      "summarizePaperCycleLifecycle",
      "EMPTY_PAPER_CYCLE_LIFECYCLE_SUMMARY",
    ];
    for (const sym of surface) {
      const lower = sym.toLowerCase();
      assertTrue(!lower.includes("submit"), `${sym}: no submit`);
      assertTrue(!lower.includes("wallet"), `${sym}: no wallet`);
      assertTrue(!lower.includes("signer"), `${sym}: no signer`);
      assertTrue(!lower.includes("privatekey"), `${sym}: no privateKey`);
    }
  });
});
