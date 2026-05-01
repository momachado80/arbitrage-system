/**
 * Tests — reliability probe runner (paper-only, synthetic).
 *
 * Paper/shadow only. NO network. NO real submission.
 * Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";
import os from "os";
import path from "path";

import { runReliabilityProbes } from "../../lib/readiness/reliabilityProbeRunner";

import { assertEqual, assertTrue, describe, test } from "./_assert";

describe("tests/readiness/reliabilityProbeRunner.test.ts", () => {
  test("disabled by default → all probes report insufficient_evidence except worker-survives", () => {
    const r = runReliabilityProbes({ enabled: false });
    assertEqual(r.enabled, false, "disabled");
    assertEqual(
      r.probes.restartRecovery,
      "insufficient_evidence",
      "restartRecovery insufficient",
    );
    assertEqual(
      r.probes.idempotency,
      "insufficient_evidence",
      "idempotency insufficient",
    );
    assertEqual(
      r.probes.duplicatePrevention,
      "insufficient_evidence",
      "duplicatePrevention insufficient",
    );
    assertEqual(
      r.probes.workerSurvivesDispatcherFailure,
      "pass",
      "worker-survives is observable today",
    );
  });

  test("enabled → all synthetic probes pass", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "probes-"));
    const r = runReliabilityProbes({ enabled: true, workingDir: dir });
    assertEqual(r.enabled, true, "enabled");
    assertEqual(r.probes.restartRecovery, "pass", "restart pass");
    assertEqual(r.probes.idempotency, "pass", "idempotency pass");
    assertEqual(r.probes.duplicatePrevention, "pass", "duplicatePrevention pass");
    assertEqual(r.probes.leaseRecovery, "pass", "leaseRecovery pass");
    assertEqual(r.probes.retryFinite, "pass", "retryFinite pass");
    assertEqual(
      r.probes.permanentErrorsBecomeDead,
      "pass",
      "permanentErrorsBecomeDead pass",
    );
    assertEqual(r.probes.circuitBreaker, "pass", "circuitBreaker pass");
    assertEqual(
      r.probes.workerSurvivesDispatcherFailure,
      "pass",
      "worker-survives pass",
    );
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("restart probe uses a temp dir (does NOT touch PAPER_STATE_DIR)", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "probes-"));
    runReliabilityProbes({ enabled: true, workingDir: dir });
    // Probe wrote `restart-mock.json` in the workingDir, not in any real state dir.
    const expected = path.join(dir, "restart-mock.json");
    assertTrue(fs.existsSync(expected), "mock written in temp dir only");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("env-driven enable: ENABLE_READINESS_SYNTHETIC_PROBES=true triggers run", () => {
    const prev = process.env.ENABLE_READINESS_SYNTHETIC_PROBES;
    try {
      process.env.ENABLE_READINESS_SYNTHETIC_PROBES = "true";
      const r = runReliabilityProbes();
      assertEqual(r.enabled, true, "enabled by env");
      assertEqual(r.probes.idempotency, "pass", "probes ran");
    } finally {
      if (prev === undefined) delete process.env.ENABLE_READINESS_SYNTHETIC_PROBES;
      else process.env.ENABLE_READINESS_SYNTHETIC_PROBES = prev;
    }
  });

  test("env-driven disable: missing env keeps insufficient_evidence", () => {
    const prev = process.env.ENABLE_READINESS_SYNTHETIC_PROBES;
    try {
      delete process.env.ENABLE_READINESS_SYNTHETIC_PROBES;
      const r = runReliabilityProbes();
      assertEqual(r.enabled, false, "disabled");
      assertEqual(
        r.probes.restartRecovery,
        "insufficient_evidence",
        "still insufficient",
      );
    } finally {
      if (prev !== undefined) process.env.ENABLE_READINESS_SYNTHETIC_PROBES = prev;
    }
  });

  test("notes mention the env var when disabled", () => {
    const r = runReliabilityProbes({ enabled: false });
    assertTrue(
      r.notes.some(n => n.includes("ENABLE_READINESS_SYNTHETIC_PROBES")),
      "notes mention env var",
    );
  });
});
