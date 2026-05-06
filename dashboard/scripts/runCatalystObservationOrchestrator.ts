/**
 * Catalyst Observation Orchestrator — automação cloud read-only.
 *
 * Substitui a sequência manual buildPlan → runDue (dry-run). Sem `.paper`, sem worker de execução,
 * sem ordens ou estado económico de paper trading.
 *
 * Uso:
 *   CATALYST_OBSERVATION_AUTOMATION_ENABLED=1 npm run observe:catalysts
 *   npm run observe:catalysts -- --once   # uma volta única (ignora ENABLED se usar --once)
 */

import { spawnSync } from "child_process";
import fs from "fs";
import path from "path";

import {
  assertSafeOrchestratorPath,
  buildDueObservationCliArgs,
  collectDedupeKeysFromLedger,
  computeDedupeKeysEmitted,
  notifyFreshDueWindows,
  parseOrchestratorEnv,
  partitionFreshDueWindows,
  summarizeDueWindows,
  type DueWindowSummaryInput,
  type OrchestratorEnvConfig,
  type OrchestratorResultClass,
} from "../lib/catalystObservationOrchestratorHelpers";

const DASHBOARD_DIR = path.resolve(__dirname, "..");

function parseArgv(argv: string[]): { once: boolean } {
  return { once: argv.includes("--once") };
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function spawnDashboard(scriptRelative: string, extraArgs: string[]): {
  ok: boolean;
  status: number | null;
  stderrTail: string;
} {
  const args = ["ts-node", "-P", "tsconfig.worker.json", scriptRelative, ...extraArgs];
  const res = spawnSync("npx", args, {
    cwd: DASHBOARD_DIR,
    encoding: "utf8",
    env: process.env,
    maxBuffer: 48 * 1024 * 1024,
  });
  const stderr = res.stderr ?? "";
  return {
    ok: res.status === 0,
    status: res.status,
    stderrTail: stderr.length > 4000 ? `${stderr.slice(0, 4000)}…` : stderr,
  };
}

function randomRunId(): string {
  return `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function validatePaths(cfg: OrchestratorEnvConfig): void {
  assertSafeOrchestratorPath(cfg.ledgerPath, "ledger");
  assertSafeOrchestratorPath(cfg.planPath, "plan");
  assertSafeOrchestratorPath(cfg.duePath, "due");
}

function ensureLedgerDir(ledgerPath: string): void {
  fs.mkdirSync(path.dirname(path.resolve(ledgerPath)), { recursive: true });
}

function appendLedgerLine(obj: Record<string, unknown>, ledgerPath: string): void {
  ensureLedgerDir(ledgerPath);
  fs.appendFileSync(path.resolve(ledgerPath), `${JSON.stringify(obj)}\n`, "utf8");
}

async function runSingleCycle(cfg: OrchestratorEnvConfig): Promise<void> {
  validatePaths(cfg);
  const runId = randomRunId();
  const timestamp = new Date().toISOString();
  const errors: string[] = [];

  const planArgs = [
    "--limit",
    String(cfg.limit),
    "--horizon-hours",
    String(cfg.horizonHours),
    "--out",
    cfg.planPath,
  ];
  const planRun = spawnDashboard("scripts/buildCatalystObservationPlan.ts", planArgs);
  if (!planRun.ok) errors.push(`plan_spawn_exit_${planRun.status ?? "null"}`);

  /** Dry-run obrigatório — modo apenas read-only. Ignora env CATALYST_OBSERVATION_DRY_RUN=false. */
  const dueArgs = buildDueObservationCliArgs(cfg);
  const dueRun = spawnDashboard("scripts/runDueObservationScouts.ts", dueArgs);
  if (!dueRun.ok) errors.push(`due_spawn_exit_${dueRun.status ?? "null"}`);

  let resultClass: OrchestratorResultClass = "NO_DUE_WINDOWS";
  let dueWindowsRaw: DueWindowSummaryInput[] = [];

  try {
    const rawDue = fs.readFileSync(path.resolve(cfg.duePath), "utf8");
    const parsed = JSON.parse(rawDue) as { dueWindows?: DueWindowSummaryInput[] };
    dueWindowsRaw = Array.isArray(parsed.dueWindows) ? parsed.dueWindows : [];
  } catch {
    errors.push("due_json_read_failed");
    resultClass = "READ_FAILED";
  }

  if (!planRun.ok) resultClass = "PLAN_FAILED";
  else if (!dueRun.ok && resultClass !== "READ_FAILED") resultClass = "DUE_FAILED";
  else if (errors.some(e => e === "due_json_read_failed")) {
    /* mantém READ_FAILED */
  } else if (dueWindowsRaw.length === 0) resultClass = "NO_DUE_WINDOWS";
  else resultClass = "HAS_DUE_WINDOWS";

  const seen = collectDedupeKeysFromLedger(cfg.ledgerPath);
  const { freshWindows, freshKeys } = partitionFreshDueWindows(dueWindowsRaw, seen);

  const { notificationSent } = await notifyFreshDueWindows({
    notifyWebhookUrl: cfg.notifyWebhookUrl,
    freshWindows,
    utcNowIso: timestamp,
  });
  if (cfg.notifyWebhookUrl && freshKeys.length > 0 && !notificationSent) {
    errors.push("webhook_all_attempts_failed");
  }

  const dedupeKeysEmitted = computeDedupeKeysEmitted({
    notifyWebhookUrl: cfg.notifyWebhookUrl,
    notificationSent,
    freshKeys,
  });

  const ledgerEntry = {
    timestamp,
    runId,
    planPath: cfg.planPath,
    duePath: cfg.duePath,
    horizonHours: cfg.horizonHours,
    dueWithinMinutes: cfg.dueWithinMinutes,
    pollIntervalSeconds: cfg.pollIntervalSeconds,
    dueWindowCount: dueWindowsRaw.length,
    dueWindows: summarizeDueWindows(dueWindowsRaw),
    resultClass,
    notificationSent,
    notifiedWindowKeys: notificationSent ? freshKeys : [],
    dedupeKeysEmitted,
    errors: errors.length ? errors : null,
    executionMode: "read_only_dry_run_forced",
    paperState: "untouched",
  };

  appendLedgerLine(ledgerEntry, cfg.ledgerPath);

  process.stdout.write(
    `[catalyst-orchestrator] ${timestamp} runId=${runId} result=${resultClass} due=${dueWindowsRaw.length} fresh=${freshKeys.length} notifications=${notificationSent}\n`,
  );

  if (freshKeys.length > 0) {
    for (const w of freshWindows) {
      process.stdout.write(
        `[catalyst-orchestrator] fresh_window market=${w.marketId} window=${w.windowType} at=${w.runAtUtc}\n`,
      );
    }
  }

  if (errors.length > 0) {
    process.stderr.write(`[catalyst-orchestrator] warnings: ${errors.join("; ")}\n`);
    if (planRun.stderrTail.trim())
      process.stderr.write(`[catalyst-orchestrator][plan stderr] ${planRun.stderrTail}\n`);
    if (dueRun.stderrTail.trim())
      process.stderr.write(`[catalyst-orchestrator][due stderr] ${dueRun.stderrTail}\n`);
  }
}

async function main(): Promise<void> {
  const { once } = parseArgv(process.argv);
  const cfg = parseOrchestratorEnv(process.env);

  if (!once && !cfg.automationEnabled) {
    process.stderr.write(
      "[catalyst-orchestrator] disabled — definir CATALYST_OBSERVATION_AUTOMATION_ENABLED=1 ou usar `--once`.\n",
    );
    process.exit(0);
  }

  process.stdout.write(
    `[catalyst-orchestrator] starting automation=${cfg.automationEnabled || once} once=${once} pollSeconds=${cfg.pollIntervalSeconds} horizonHours=${cfg.horizonHours} dryRunForced=true ledger=${cfg.ledgerPath}\n`,
  );

  validatePaths(cfg);

  let shuttingDown = false;
  process.on("SIGINT", () => {
    shuttingDown = true;
    process.stderr.write("\n[catalyst-orchestrator] SIGINT → graceful stop\n");
  });
  process.on("SIGTERM", () => {
    shuttingDown = true;
    process.stderr.write("\n[catalyst-orchestrator] SIGTERM → graceful stop\n");
  });

  do {
    await runSingleCycle(cfg);
    if (once || shuttingDown) break;
    await sleep(cfg.pollIntervalSeconds * 1000);
  } while (!shuttingDown);

  process.stdout.write("[catalyst-orchestrator] exit_ok\n");
}

main().catch(err => {
  process.stderr.write(`[catalyst-orchestrator] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
