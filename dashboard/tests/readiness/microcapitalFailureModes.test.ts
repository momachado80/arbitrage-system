/**
 * Tests — adversarial failure modes for the readiness gate.
 *
 * These probes use SYNTHETIC reliability inputs to confirm the analyzer reacts
 * appropriately. They never touch real dispatchers, queues, signers, wallets,
 * or order endpoints. Paper/shadow only — policy: shadow_only_no_api_submit_v1.
 *
 * Coverage:
 *   A. Restart with queued packets
 *   B. In-flight with expired lease
 *   C. Timeout / retry / dead
 *   D. Permanent errors → dead
 *   E. Circuit breaker
 *   F. Worker survives dispatcher failure
 *   G. Safety payload (apiSubmitAllowed=false, intendedAction=audit_only,
 *      mode=shadow_only, magnitude.unit=validation_only, magnitude.value=1)
 */

import {
  analyzeMicrocapitalReadiness,
  type AnalyzerInput,
  type ReliabilityProbeInput,
} from "../../lib/readiness/analyzeMicrocapitalReadiness";
import { isBlockedVerdict } from "../../lib/readiness/microcapitalReadinessDossier";

import {
  assertEqual,
  assertFalse,
  assertThrows,
  assertTrue,
  describe,
  test,
} from "./_assert";

import { assertShadowOnlyNoRealSubmission } from "../../lib/crossVenueAnchor1823789ShadowMicroLive";

const ALL_PASS_RELIABILITY: ReliabilityProbeInput = {
  restartRecoveryPassed: true,
  idempotencyPassed: true,
  duplicatePreventionPassed: true,
  leaseRecoveryPassed: true,
  retryFinitePassed: true,
  permanentErrorsBecomeDeadPassed: true,
  circuitBreakerPassed: true,
  workerSurvivesDispatcherFailurePassed: true,
};

function baseInput(reliability: Partial<ReliabilityProbeInput> = {}): AnalyzerInput {
  return {
    systemIdentity: {
      projectName: "test",
      generatedAt: "2026-04-30T00:00:00.000Z",
      gitCommit: null,
      paperStateDir: "/tmp/.paper",
      executionMode: "controlled_paper",
      effectiveMode: "shadow_only",
      workerName: "test",
    },
    paperAnalytics: null,
    closedTrades: [],
    activeTrades: [],
    realismSamples: [],
    prohibitedTermsScan: { scannedFileCount: 1, findings: [], passed: true },
    apiSubmitAllowedTrueAnywhere: false,
    dispatcherAuditOnly: true,
    payloadSafetyPresent: true,
    reliability: { ...ALL_PASS_RELIABILITY, ...reliability },
    observability: {
      hasJsonLogs: true,
      hasShadowAudit: true,
      hasPaperTrades: true,
      hasAnalyticsEndpoint: true,
      hasSystemEndpoint: true,
      hasTradesEndpoint: true,
      canReconcileCycleToPacket: true,
      canReconcilePacketToDispatch: true,
      canReconcileSignalToPaperTrade: true,
    },
    riskLimits: {
      killSwitchSimulated: true,
      cooldownAfterLossSimulated: true,
      exposureLimitSimulated: true,
      maxConsecutiveNegativeCycles: 0,
      maxPaperLossPerCycle: 0,
      maxPaperDailyLoss: 0,
    },
  };
}

describe("tests/readiness/microcapitalFailureModes.test.ts", () => {
  // A. Restart
  test("A. restart with queued packet — failure flagged as technical block", () => {
    const d = analyzeMicrocapitalReadiness(baseInput({ restartRecoveryPassed: false }));
    assertTrue(
      isBlockedVerdict(d.reliabilityReadiness.reliabilityVerdict),
      "restart-recovery failure must block reliability",
    );
  });

  test("A. restart-recovery pass keeps reliability green", () => {
    const d = analyzeMicrocapitalReadiness(baseInput());
    assertEqual(
      d.reliabilityReadiness.reliabilityVerdict,
      "APPROVED_FOR_SHADOW_READINESS",
      "all-pass reliability is approved",
    );
  });

  // B. In-flight lease
  test("B. lease-recovery failure surfaces as warning (no block)", () => {
    const d = analyzeMicrocapitalReadiness(baseInput({ leaseRecoveryPassed: false }));
    assertEqual(
      d.reliabilityReadiness.reliabilityVerdict,
      "APPROVED_WITH_REMARKS",
      "lease failure is a remark",
    );
  });

  // C. Timeout / retry
  test("C. retry-finite failure surfaces in notes", () => {
    const d = analyzeMicrocapitalReadiness(baseInput({ retryFinitePassed: false }));
    assertEqual(
      d.reliabilityReadiness.retryFinitePassed,
      false,
      "retryFinite=false propagated",
    );
    assertTrue(
      d.reliabilityReadiness.notes.some(n => n.includes("retryFinite")),
      "retryFinite mentioned in notes",
    );
  });

  // D. Permanent errors → dead
  test("D. permanent errors must become dead (probe flag honored)", () => {
    const d = analyzeMicrocapitalReadiness(
      baseInput({ permanentErrorsBecomeDeadPassed: false }),
    );
    assertEqual(
      d.reliabilityReadiness.permanentErrorsBecomeDeadPassed,
      false,
      "flag propagated",
    );
  });

  // E. Circuit breaker
  test("E. circuit-breaker failure surfaces in notes", () => {
    const d = analyzeMicrocapitalReadiness(baseInput({ circuitBreakerPassed: false }));
    assertTrue(
      d.reliabilityReadiness.notes.some(n => n.includes("circuitBreaker")),
      "circuit breaker mentioned",
    );
  });

  // F. Worker survives dispatcher failure
  test("F. worker-survives-dispatcher-failure flag propagates", () => {
    const d = analyzeMicrocapitalReadiness(
      baseInput({ workerSurvivesDispatcherFailurePassed: false }),
    );
    assertEqual(
      d.reliabilityReadiness.workerSurvivesDispatcherFailurePassed,
      false,
      "flag propagated",
    );
  });

  // G. Safety payload
  test("G. safety payload — assertShadowOnlyNoRealSubmission passes for safe values", () => {
    const safe = {
      apiSubmitAllowed: false,
      intendedAction: "audit_only",
      mode: "shadow_only",
      magnitude: { unit: "validation_only", value: 1 },
      realSubmission: false,
      wouldSubmit: false,
      shadowRealSubmission: false,
    };
    assertEqual(safe.apiSubmitAllowed, false, "apiSubmitAllowed must be false");
    assertEqual(safe.intendedAction, "audit_only", "intendedAction must be audit_only");
    assertEqual(safe.mode, "shadow_only", "mode must be shadow_only");
    assertEqual(safe.magnitude.unit, "validation_only", "magnitude.unit");
    assertEqual(safe.magnitude.value, 1, "magnitude.value");
    // The defensive guard must not throw on safe payloads.
    assertShadowOnlyNoRealSubmission(safe);
  });

  test("G. safety payload — guard throws if realSubmission flips to true", () => {
    assertThrows(
      () =>
        assertShadowOnlyNoRealSubmission({
          realSubmission: true,
          wouldSubmit: false,
          shadowRealSubmission: false,
        }),
      "guard must throw on realSubmission=true",
    );
    assertThrows(
      () =>
        assertShadowOnlyNoRealSubmission({
          realSubmission: false,
          wouldSubmit: true,
          shadowRealSubmission: false,
        }),
      "guard must throw on wouldSubmit=true",
    );
    assertThrows(
      () =>
        assertShadowOnlyNoRealSubmission({
          realSubmission: false,
          wouldSubmit: false,
          shadowRealSubmission: true,
        }),
      "guard must throw on shadowRealSubmission=true",
    );
  });

  test("G. safety payload — guard does not throw for absent fields (no-op safe)", () => {
    assertShadowOnlyNoRealSubmission({});
    assertFalse(false, "no throw means pass");
  });
});
