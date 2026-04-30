/**
 * Microcapital Readiness Gate — thresholds.
 *
 * IMPORTANT: "microcapital" here means *technical readiness simulated on paper*.
 * It does NOT authorize real execution, capital activation, or financial submit.
 * Policy preserved: shadow_only_no_api_submit_v1.
 *
 * Conservative defaults; configurable only by editing this file.
 * Do NOT bind these to environment variables — preventing dangerous overrides.
 */

export const SHADOW_ONLY_POLICY_TAG = "shadow_only_no_api_submit_v1" as const;

export const MICROCAPITAL_READINESS_THRESHOLDS = {
  /** Minimum number of paper-observed cycles before economic statements are credible. */
  MIN_OBSERVED_CYCLES: 100,
  /** Minimum positive cycle rate (0..1) considered economically meaningful. */
  MIN_POSITIVE_CYCLE_RATE: 0.6,
  /** Minimum net edge after fees + slippage, expressed in paper-USD. */
  MIN_NET_AFTER_FEES_AND_SLIPPAGE: 0,
  /** Maximum paper drawdown allowed (absolute, paper-USD). Computed against starting paper capital. */
  MAX_DRAWDOWN_PAPER_ALLOWED_ABS: 1_500,
  /** Maximum stale signal rate (0..1). */
  MAX_STALE_SIGNAL_RATE: 0.1,
  /** Maximum allowed duplicate packet count detected in dispatcher state. */
  MAX_DUPLICATE_PACKET_COUNT: 0,
  /** Maximum allowed unrecovered in-flight packets after restart. */
  MAX_UNRECOVERED_IN_FLIGHT_PACKETS: 0,
  /** Pass rate required for restart-recovery test (0..1). */
  MIN_RESTART_RECOVERY_PASS_RATE: 1,
  /** Pass rate required for idempotency test (0..1). */
  MIN_IDEMPOTENCY_PASS_RATE: 1,
  /** Concentration threshold: share of total net edge held by top-1 cycle (0..1). */
  MAX_TOP_CYCLE_EDGE_SHARE: 0.5,
  /** Concentration threshold: share of total net edge held by top-3 cycles (0..1). */
  MAX_TOP_3_CYCLES_EDGE_SHARE: 0.85,
  /** Default fee rate applied per leg in simulated execution realism (paper-only). */
  DEFAULT_PAPER_FEE_RATE: 0.002,
  /** Default conservative slippage in basis points applied to paper realism. */
  DEFAULT_PAPER_SLIPPAGE_BPS: 25,
  /** Latency budgets (ms) for signal-to-enqueue path. */
  MAX_AVG_SIGNAL_TO_ENQUEUE_MS: 250,
  MAX_P95_SIGNAL_TO_ENQUEUE_MS: 750,
  /** Latency budgets (ms) for enqueue-to-dispatch path. */
  MAX_AVG_DISPATCH_LATENCY_MS: 500,
  MAX_P95_DISPATCH_LATENCY_MS: 1_500,
} as const;

export type MicrocapitalReadinessThresholds = typeof MICROCAPITAL_READINESS_THRESHOLDS;

/**
 * List of terms that MUST NOT appear in production source files (lib/**, scripts/**,
 * app/api/**, workers/**) outside the explicit whitelist. Tests check this scanner.
 *
 * NOTE: this constant intentionally lists the terms as data so the readiness
 * scanner is whitelisted: the scanner itself is allowed to mention them.
 */
export const PROHIBITED_EXECUTION_TERMS: ReadonlyArray<string> = [
  "submitOrder",
  "placeOrder",
  "executeTrade",
  "realSubmit",
  "apiSubmitAllowed: true",
  "microLiveSubmit",
  "externalOrderEndpoint",
  "privateKey",
  "wallet",
  "signer",
  "transaction",
];

/**
 * Files explicitly allowed to mention prohibited terms because they are themselves
 * scanners, audit guards, readiness reports, or paper-only safety checks.
 *
 * Path matching is suffix-based against the absolute file path.
 */
export const PROHIBITED_TERMS_WHITELIST: ReadonlyArray<string> = [
  // Existing protected core scanner.
  "scripts/assertCrossVenueAnchor1823789ShadowOnly.ts",
  // Readiness layer (this PR).
  "lib/readiness/microcapitalReadinessThresholds.ts",
  "lib/readiness/microcapitalReadinessDossier.ts",
  "lib/readiness/analyzeMicrocapitalReadiness.ts",
  "lib/readiness/executionRealismHarness.ts",
  "lib/readiness/renderMicrocapitalReadinessExecutiveSummary.ts",
  "lib/readiness/prohibitedTermsScanner.ts",
  "lib/readiness/composeMicrocapitalReadinessDossier.ts",
  "app/api/readiness/microcapital/route.ts",
  "scripts/readiness/microcapital-readiness.ts",
  "tests/readiness/microcapitalReadinessDossier.test.ts",
  "tests/readiness/executionRealismHarness.test.ts",
  "tests/readiness/microcapitalFailureModes.test.ts",
  "tests/readiness/prohibitedExecutionTerms.test.ts",
  "tests/readiness/run-readiness-tests.ts",
];
