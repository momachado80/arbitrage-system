/**
 * Helpers read-only para o Catalyst Observation Orchestrator (cloud automation).
 * Sem rede obrigatória nos parsers; sem `.paper`; sem execução económica.
 */

import fs from "fs";
import path from "path";

export interface DueWindowSummaryInput {
  marketId?: string | null;
  label?: string | null;
  windowType?: string | null;
  runAtUtc?: string | null;
  reason?: string | null;
}

export type OrchestratorResultClass =
  | "NO_DUE_WINDOWS"
  | "HAS_DUE_WINDOWS"
  | "READ_FAILED"
  | "PLAN_FAILED"
  | "DUE_FAILED";

export interface OrchestratorEnvConfig {
  automationEnabled: boolean;
  horizonHours: number;
  limit: number;
  dueWithinMinutes: number;
  pollIntervalSeconds: number;
  dryRun: boolean;
  ledgerPath: string;
  planPath: string;
  duePath: string;
  notifyWebhookUrl: string | null;
}

const SCOUT_TYPE_DRY = "catalyst_scout_dry";

export function assertSafeOrchestratorPath(absPath: string, label: string): void {
  const norm = path.resolve(absPath);
  if (norm.includes(`${path.sep}.paper${path.sep}`) || norm.endsWith(`${path.sep}.paper`)) {
    throw new Error(`${label}_path_blocked:.paper`);
  }
}

function envBool(raw: string | undefined, defaultVal: boolean): boolean {
  if (raw === undefined || raw === "") return defaultVal;
  const v = raw.trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

function envInt(raw: string | undefined, defaultVal: number, min: number): number {
  if (raw === undefined || raw === "") return defaultVal;
  const n = parseInt(raw.trim(), 10);
  return Number.isFinite(n) && n >= min ? n : defaultVal;
}

export function parseOrchestratorEnv(env: NodeJS.ProcessEnv): OrchestratorEnvConfig {
  return {
    automationEnabled: envBool(env.CATALYST_OBSERVATION_AUTOMATION_ENABLED, false),
    horizonHours: envInt(env.CATALYST_OBSERVATION_HORIZON_HOURS, 72, 1),
    limit: envInt(env.CATALYST_OBSERVATION_LIMIT, 8, 1),
    dueWithinMinutes: envInt(env.CATALYST_OBSERVATION_DUE_WITHIN_MINUTES, 15, 1),
    pollIntervalSeconds: envInt(env.CATALYST_OBSERVATION_POLL_INTERVAL_SECONDS, 300, 5),
    dryRun: envBool(env.CATALYST_OBSERVATION_DRY_RUN, true),
    ledgerPath: env.CATALYST_OBSERVATION_LEDGER_PATH ?? "/data/catalyst-observation-ledger.jsonl",
    planPath: env.CATALYST_OBSERVATION_PLAN_PATH ?? "/tmp/catalyst-observation-plan.json",
    duePath: env.CATALYST_OBSERVATION_DUE_PATH ?? "/tmp/due-observation-scouts.json",
    notifyWebhookUrl:
      typeof env.CATALYST_OBSERVATION_NOTIFY_WEBHOOK_URL === "string" &&
      env.CATALYST_OBSERVATION_NOTIFY_WEBHOOK_URL.trim()
        ? env.CATALYST_OBSERVATION_NOTIFY_WEBHOOK_URL.trim()
        : null,
  };
}

export function buildWindowKey(w: DueWindowSummaryInput): string {
  const mid = String(w.marketId ?? "").trim();
  const t = String(w.runAtUtc ?? "").trim();
  const wt = String(w.windowType ?? "").trim();
  return `${mid}|${t}|${wt}|${SCOUT_TYPE_DRY}`;
}

export function summarizeDueWindows(dueWindows: DueWindowSummaryInput[]): Array<Record<string, unknown>> {
  return dueWindows.map(w => ({
    marketId: w.marketId ?? null,
    label:
      typeof w.label === "string"
        ? w.label.length > 140
          ? `${w.label.slice(0, 140)}…`
          : w.label
        : null,
    windowType: w.windowType ?? null,
    runAtUtc: w.runAtUtc ?? null,
    windowKey: buildWindowKey(w),
  }));
}

export function collectDedupeKeysFromLedger(ledgerPath: string, maxLinesToScan = 8000): Set<string> {
  const seen = new Set<string>();
  if (!ledgerPath || !fs.existsSync(ledgerPath)) return seen;
  try {
    const raw = fs.readFileSync(ledgerPath, "utf8");
    const lines = raw.split(/\r?\n/).filter(Boolean);
    const slice = lines.length > maxLinesToScan ? lines.slice(-maxLinesToScan) : lines;
    for (const line of slice) {
      try {
        const j = JSON.parse(line) as Record<string, unknown>;
        const emitted = j.dedupeKeysEmitted;
        const notified = j.notifiedWindowKeys;
        for (const src of [emitted, notified]) {
          if (!Array.isArray(src)) continue;
          for (const k of src) seen.add(String(k));
        }
      } catch {
        /* linha inválida → ignorar */
      }
    }
  } catch {
    /* ledger ausente ou ilegível → conjunto vazio */
  }
  return seen;
}

export function partitionFreshDueWindows(
  dueWindows: DueWindowSummaryInput[],
  seenKeys: Set<string>,
): { freshWindows: DueWindowSummaryInput[]; freshKeys: string[] } {
  const freshWindows: DueWindowSummaryInput[] = [];
  const freshKeys: string[] = [];
  for (const w of dueWindows) {
    const k = buildWindowKey(w);
    if (seenKeys.has(k)) continue;
    freshWindows.push(w);
    freshKeys.push(k);
  }
  return { freshWindows, freshKeys };
}

export function formatSaoPauloLocal(isoUtc: string): string {
  const d = new Date(isoUtc.trim());
  if (!Number.isFinite(d.getTime())) return isoUtc;
  return new Intl.DateTimeFormat("pt-BR", {
    timeZone: "America/Sao_Paulo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(d);
}

export function buildWebhookPayload(input: {
  window: DueWindowSummaryInput;
  utcNowIso: string;
  status: string;
  recommendedNextState: string;
  severity?: string;
}): Record<string, unknown> {
  const title = "Catalyst observation window due (dry-run)";
  const market =
    (typeof input.window.label === "string" && input.window.label.trim()) ||
    String(input.window.marketId ?? "unknown_market");
  const windowBits = `${input.window.windowType ?? "?"} @ ${input.window.runAtUtc ?? "?"}`;
  return {
    title,
    severity: input.severity ?? "info",
    market,
    window: windowBits,
    localTimeAmericaSaoPaulo: formatSaoPauloLocal(input.utcNowIso),
    utcTime: input.utcNowIso,
    status: input.status,
    recommendedNextState: input.recommendedNextState,
    execution: "none",
    paperState: "untouched",
  };
}

export function computeDedupeKeysEmitted(opts: {
  notifyWebhookUrl: string | null;
  notificationSent: boolean;
  freshKeys: string[];
}): string[] {
  if (opts.notifyWebhookUrl && !opts.notificationSent) return [];
  return [...opts.freshKeys];
}

export function buildDueObservationCliArgs(
  cfg: Pick<OrchestratorEnvConfig, "planPath" | "duePath" | "dueWithinMinutes">,
): string[] {
  return [
    "--plan",
    cfg.planPath,
    "--due-within-minutes",
    String(cfg.dueWithinMinutes),
    "--out",
    cfg.duePath,
    "--dry-run",
    "true",
  ];
}

/** Uma tentativa POST por janela fresh; acumula sucesso para dedupe (`notificationSent`). */
export async function notifyFreshDueWindows(opts: {
  notifyWebhookUrl: string | null;
  freshWindows: DueWindowSummaryInput[];
  utcNowIso: string;
  fetchFn?: typeof fetch;
}): Promise<{ notificationSent: boolean }> {
  if (!opts.notifyWebhookUrl || opts.freshWindows.length === 0) {
    return { notificationSent: false };
  }
  const fetchImpl = opts.fetchFn ?? fetch;
  let notificationSent = false;
  for (const w of opts.freshWindows) {
    const body = buildWebhookPayload({
      window: w,
      utcNowIso: opts.utcNowIso,
      status: "due_window_detected_dry_run",
      recommendedNextState: "manual_review_then_optional_scout_when_POLICY_ALLOWS",
    });
    try {
      const res = await fetchImpl(opts.notifyWebhookUrl, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(12_000),
      });
      notificationSent = notificationSent || res.ok;
    } catch {
      /* falha de rede → retry no ciclo seguinte se dedupe não consumir chaves */
    }
  }
  return { notificationSent };
}
