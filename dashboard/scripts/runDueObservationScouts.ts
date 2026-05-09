/**
 * Agrupa janelas do plano de catalisador perto do instante atual e executa scout read-only quando habilitado.
 * Default `--dry-run true`: não executa scout nem worker/watcher.
 * `--scout-readonly true`: roda scout read-only mesmo em dry-run, gerando WINDOW_INFORMATIVE.
 *
 * GARANTIAS DE SEGURANÇA quando scout é executado:
 *  - o spawn é exclusivamente `scripts/scoutObservationWindows.ts` (verificado via fs.existsSync antes do spawn)
 *  - scoutObservationWindows.ts é read-only por contrato (auditado em check:shadow-only)
 *  - early return abort se o script alvo não existir, evitando spawn de binário arbitrário
 *  - SHADOW_ONLY=1 propagado no env para defense-in-depth
 */

import { spawnSync } from "child_process";
import fs from "fs";
import path from "path";

const REPO_ABS = path.resolve(__dirname, "..", "..");

interface ObservationWindowSlot {
  windowType: string;
  runAtUtc: string;
  reason: string;
}

interface PlanRow {
  marketId: string;
  label?: string | null;
  catalystReadinessVerdict?: string;
  observationWindows?: ObservationWindowSlot[];
}

interface PlanFile {
  plan?: PlanRow[];
}

function assertOutPath(absPath: string): void {
  const norm = path.resolve(absPath);
  if (norm.includes(`${path.sep}.paper${path.sep}`) || norm.endsWith(`${path.sep}.paper`)) {
    throw new Error("output_path_blocked:.paper");
  }
  const inside = norm === REPO_ABS || norm.startsWith(REPO_ABS + path.sep);
  const tmpOk =
    norm === "/tmp" ||
    norm.startsWith("/tmp/") ||
    norm === "/private/tmp" ||
    norm.startsWith("/private/tmp/");
  if (!(tmpOk || !inside)) throw new Error("output_must_be_tmp_or_outside_repo");
}

function parseBool(argv: string[], flag: string, defaultVal: boolean): boolean {
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === flag && argv[i + 1]) {
      const v = String(argv[++i]).toLowerCase();
      return v === "true" || v === "1" || v === "yes";
    }
    if (a.startsWith(`${flag}=`)) {
      const v = a.slice(`${flag}=`.length).toLowerCase();
      return v === "true" || v === "1" || v === "yes";
    }
  }
  return defaultVal;
}

function parseCli(argv: string[]): {
  planPath: string;
  dueWithinMinutes: number;
  outPath: string;
  dryRun: boolean;
  scoutReadonly: boolean;
} {
  let planPath = "/tmp/catalyst-observation-plan.json";
  let dueWithinMinutes = 15;
  let outPath = "/tmp/due-observation-scouts.json";
  const dryRun = parseBool(argv, "--dry-run", true);
  const scoutReadonly = parseBool(argv, "--scout-readonly", false);

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--plan" && argv[i + 1]) planPath = argv[++i]!;
    else if (a.startsWith("--plan=")) planPath = a.slice("--plan=".length);
    else if (a === "--due-within-minutes" && argv[i + 1])
      dueWithinMinutes = Math.max(1, parseInt(argv[++i]!, 10) || 15);
    else if (a.startsWith("--due-within-minutes="))
      dueWithinMinutes = Math.max(1, parseInt(a.slice("--due-within-minutes=".length), 10) || 15);
    else if (a === "--out" && argv[i + 1]) outPath = argv[++i]!;
    else if (a.startsWith("--out=")) outPath = a.slice("--out=".length);
  }

  return { planPath, dueWithinMinutes, outPath, dryRun, scoutReadonly };
}

function truncStrLocal(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
}

function parseUtcMs(iso: string): number | null {
  const t = new Date(iso.trim()).getTime();
  return Number.isFinite(t) ? t : null;
}

function pickDueWindows(plan: PlanFile, nowMs: number, dueWithinMinutes: number): Array<
  Record<string, unknown>
> {
  const due: Array<Record<string, unknown>> = [];
  const horizonMs = nowMs + dueWithinMinutes * 60_000;
  const pastGraceMs = nowMs - 10 * 60_000;

  for (const row of plan.plan ?? []) {
    const wins = row.observationWindows ?? [];
    for (const w of wins) {
      const ts = parseUtcMs(w.runAtUtc);
      if (ts === null) continue;
      if (ts <= horizonMs && ts >= pastGraceMs) {
        due.push({
          marketId: row.marketId,
          label: row.label ?? null,
          catalystReadinessVerdict: row.catalystReadinessVerdict ?? null,
          windowType: w.windowType,
          runAtUtc: w.runAtUtc,
          reason: w.reason,
        });
      }
    }
  }
  due.sort((a, b) => String(a.runAtUtc).localeCompare(String(b.runAtUtc)));
  return due;
}

function scoutCommandParts(tmpOut: string): string[] {
  return [
    "npx",
    "ts-node",
    "-P",
    "tsconfig.worker.json",
    "scripts/scoutObservationWindows.ts",
    "--universe",
    "clean",
    "--limit",
    "8",
    "--snapshots",
    "6",
    "--gap-seconds",
    "60",
    "--out",
    tmpOut,
  ];
}

function main(): void {
  const cli = parseCli(process.argv);
  const generatedAtUtc = new Date().toISOString();
  const nowMs = Date.parse(generatedAtUtc);

  const rawPlan = fs.readFileSync(path.resolve(cli.planPath), "utf8");
  const plan = JSON.parse(rawPlan) as PlanFile;

  const dueWindows = pickDueWindows(plan, nowMs, cli.dueWithinMinutes);
  const tmpOut = `/tmp/observation-scout-triggered-${Date.now()}.json`;
  const cmdParts = scoutCommandParts(tmpOut);
  const shellCmd =
    dueWindows.length === 0
      ? ""
      : `cd dashboard && ${cmdParts.map(x => (/\s/.test(x) ? JSON.stringify(x) : x)).join(" ")}`;
  const wouldRunCommands = shellCmd ? [shellCmd] : [];

  const executedScouts: Array<{
    exited: number;
    outJson?: string;
    stderrTail?: string;
    mode: string;
  }> = [];

  const shouldRunScout = (!cli.dryRun || cli.scoutReadonly) && dueWindows.length > 0;

  if (shouldRunScout) {
    /**
     * SAFETY GATE: confirma que o alvo do spawn é o scout read-only auditado.
     * Se o caminho do script estiver ausente, recusa rodar — evita spawn de binário arbitrário.
     */
    const scoutTarget = path.join(REPO_ABS, "dashboard", "scripts", "scoutObservationWindows.ts");
    if (!fs.existsSync(scoutTarget)) {
      throw new Error("scout_target_missing:scripts/scoutObservationWindows.ts");
    }

    /**
     * SAFETY GATE: env defensivo. SHADOW_ONLY=1 sinaliza para qualquer guard downstream.
     * scoutObservationWindows.ts já é read-only por contrato (auditado em check:shadow-only),
     * mas o env reforça a intenção operacional.
     */
    const dashDir = path.join(REPO_ABS, "dashboard");
    const safeEnv = { ...process.env, SHADOW_ONLY: "1" };
    const res = spawnSync(cmdParts[0]!, cmdParts.slice(1), {
      cwd: dashDir,
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
      env: safeEnv,
    });
    executedScouts.push({
      exited: res.status ?? 1,
      outJson: tmpOut,
      stderrTail: res.stderr ? truncStrLocal(res.stderr, 2000) : undefined,
      mode: cli.scoutReadonly && cli.dryRun ? "scout_readonly_dry_run" : "scout_legacy_non_dry_run",
    });
  }

  const payload = {
    generatedAtUtc,
    dueWithinMinutes: cli.dueWithinMinutes,
    dryRun: cli.dryRun,
    scoutReadonly: cli.scoutReadonly,
    dueWindows,
    wouldRunCommands,
    executedScouts,
    note: shouldRunScout
      ? "scout_executed_read_only_no_paper_no_orders_window_informative_possible"
      : "due_runner_prefers_single_clean_scout_when_any_window_due_no_track_choice_without_WINDOW_INFORMATIVE",
  };

  const text = `${JSON.stringify(payload, null, 2)}\n`;
  process.stdout.write(text);
  assertOutPath(cli.outPath);
  fs.mkdirSync(path.dirname(path.resolve(cli.outPath)), { recursive: true });
  fs.writeFileSync(path.resolve(cli.outPath), text, "utf8");
}

main();
