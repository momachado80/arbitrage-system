#!/usr/bin/env node
/**
 * Runtime snapshot: fetches audit from production (or AUDIT_URL) and saves consolidated JSON.
 *
 * Usage: npm run shadow:snapshot  (from dashboard/)
 * Output: reports/runtime_snapshot_latest.json
 */

import * as fs from "fs";
import * as path from "path";
import { httpGet } from "./shadow-http";
import { THRESHOLDS } from "./shadow-thresholds";

const PRODUCTION_CANDIDATES = [
  "https://dashboard-next-production-b126.up.railway.app",
  "https://web-production-800bb.up.railway.app",
  "https://web-production-eca1e.up.railway.app",
];
const EXPECTED_PROFILE_ID = "shadow_1000_structural_riskmanaged_v1";
const EXIT_KILL_PROFILE_ID = "shadow_1000_structural_exitkill_v1";
const EXIT_KILL_WINDOW180_PROFILE_ID = "shadow_1000_structural_exitkill_window180_v1";
const LATE_EXIT_PROFILE_ID = "shadow_1000_structural_lateexit_nonreversion_v1";
const FETCH_TIMEOUT_MS = 15000;
const SNAPSHOT_PATH = "reports/runtime_snapshot_latest.json";

async function getBaseUrl(): Promise<string> {
  let base =
    process.env.AUDIT_URL ||
    process.env.DASHBOARD_URL ||
    null;

  if (!base) {
    const urlFile = path.join(__dirname, "..", ".production-url");
    if (fs.existsSync(urlFile)) {
      base = fs.readFileSync(urlFile, "utf-8").trim();
    }
  }

  if (!base) {
    for (const c of PRODUCTION_CANDIDATES) {
      try {
        const res = await httpGet(`${c.replace(/\/$/, "")}/api/shadow/audit`, 5000);
        if (res.ok) {
          base = c;
          break;
        }
      } catch {
        continue;
      }
    }
  }

  if (!base) {
    throw new Error(
      "No audit URL. Set AUDIT_URL, create dashboard/.production-url, or ensure production is reachable."
    );
  }
  return base.replace(/\/$/, "");
}

interface DefenseActivation {
  structuralGateByPair: { activated: boolean; rejections: number; note: string };
  capfloor: { activated: boolean; rejections: number; note: string };
  degratio: { activated: boolean; rejections: number; note: string };
  capitalMultiplier: { activated: boolean; avgMultiplier: number; note: string };
  earlyThesisFailure: { activated: boolean; exitCount: number; note: string };
}

interface Snapshot {
  timestamp: string;
  url: string;
  economic_read_valid: boolean;
  economic_read_reason: string;
  structuralRiskManagedDiagnostics: Record<string, unknown> | null;
  structuralRiskManagedComparison_vs_shadow_1000: Record<string, unknown> | null;
  structuralRiskManagedComparison_vs_exitrefine: Record<string, unknown> | null;
  structuralExitKillDiagnostics: Record<string, unknown> | null;
  structuralExitKillComparison: Record<string, Record<string, unknown>> | null;
  structuralExitKillWindow180Diagnostics: Record<string, unknown> | null;
  structuralExitKillWindow180Comparison: Record<string, Record<string, unknown>> | null;
  structuralExitKillWindow180CausalAudit: Record<string, unknown> | null;
  structuralLateExitDiagnostics: Record<string, unknown> | null;
  structuralLateExitComparison: Record<string, Record<string, unknown>> | null;
  profileSummary: Record<string, unknown> | null;
  exitKillProfileSummary: Record<string, unknown> | null;
  exitKillWindow180ProfileSummary: Record<string, unknown> | null;
  lateExitProfileSummary: Record<string, unknown> | null;
  metrics: {
    openedTradeCount: number;
    closedTradeCount: number;
    activeTrades: number;
    avgRealizedPnL: number;
    totalRealizedPnL: number;
    avgCapitalMultiplierOpened: number;
    earlyThesisFailureExitCount: number;
    rejectedByStructuralPairCount: number;
    rejectedByCapfloorCount: number;
    rejectedByDegRatioCount: number;
  };
  defenseActivation: DefenseActivation;
}

async function run(): Promise<void> {
  const baseUrl = await getBaseUrl();
  const url = `${baseUrl}/api/shadow/audit`;

  let audit: Record<string, unknown>;
  try {
    const res = await httpGet(url, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    audit = (await res.json()) as Record<string, unknown>;
  } catch (e) {
    process.stderr.write(
      `Snapshot fetch failed: ${e instanceof Error ? e.message : String(e)}\n`
    );
    process.exit(1);
  }

  const diag = audit.structuralRiskManagedDiagnostics as Record<string, unknown> | undefined;
  const comp = audit.structuralRiskManagedComparison as Record<string, unknown> | undefined;
  const summaries = audit.profileSummaries as Array<Record<string, unknown>> | undefined;

  const compVs1000 =
    comp != null && typeof comp === "object"
      ? (comp["shadow_1000"] as Record<string, unknown>) ?? null
      : null;
  const compVsExitrefine =
    comp != null && typeof comp === "object"
      ? (comp["shadow_1000_adapt_captrade_exitrefine_v1"] as Record<string, unknown>) ?? null
      : null;

  const profileSummary =
    Array.isArray(summaries)
      ? summaries.find((p) => p?.profileId === EXPECTED_PROFILE_ID) ?? null
      : null;
  const exitKillProfileSummary =
    Array.isArray(summaries)
      ? summaries.find((p) => p?.profileId === EXIT_KILL_PROFILE_ID) ?? null
      : null;
  const exitKillWindow180ProfileSummary =
    Array.isArray(summaries)
      ? summaries.find((p) => p?.profileId === EXIT_KILL_WINDOW180_PROFILE_ID) ?? null
      : null;

  const exitKillDiag = audit.structuralExitKillDiagnostics as Record<string, unknown> | undefined;
  const exitKillComp = audit.structuralExitKillComparison as Record<string, Record<string, unknown>> | undefined;
  const exitKillWindow180Diag = audit.structuralExitKillWindow180Diagnostics as Record<string, unknown> | undefined;
  const exitKillWindow180Comp = audit.structuralExitKillWindow180Comparison as Record<string, Record<string, unknown>> | undefined;
  const exitKillWindow180Causal = audit.structuralExitKillWindow180CausalAudit as Record<string, unknown> | undefined;
  const lateExitDiag = audit.structuralLateExitDiagnostics as Record<string, unknown> | undefined;
  const lateExitComp = audit.structuralLateExitComparison as Record<string, Record<string, unknown>> | undefined;
  const lateExitProfileSummary =
    Array.isArray(summaries)
      ? summaries.find((p) => p?.profileId === LATE_EXIT_PROFILE_ID) ?? null
      : null;

  const def = (v: unknown, d: number) => (typeof v === "number" ? v : d);
  const opened = def(diag?.openedTradeCount, 0);
  const closed = profileSummary ? def(profileSummary.totalClosed, 0) : 0;
  const metrics = {
    openedTradeCount: opened,
    closedTradeCount: closed,
    activeTrades: Math.max(0, opened - closed),
    avgRealizedPnL: def(diag?.avgRealizedPnL ?? profileSummary?.avgRealizedPnL, 0),
    totalRealizedPnL: def(diag?.totalRealizedPnL ?? profileSummary?.totalRealizedPnL, 0),
    avgCapitalMultiplierOpened: def(diag?.avgCapitalMultiplierOpened, 0),
    earlyThesisFailureExitCount: def(diag?.earlyThesisFailureExitCount, 0),
    rejectedByStructuralPairCount: def(diag?.rejectedByStructuralPairCount, 0),
    rejectedByCapfloorCount: def(diag?.rejectedByCapfloorCount, 0),
    rejectedByDegRatioCount: def(diag?.rejectedByDegRatioCount, 0),
  };

  const hasDiag = diag != null && typeof diag === "object";
  const deployStale = !hasDiag;

  let economic_read_valid = false;
  let economic_read_reason: string;
  if (deployStale) {
    economic_read_reason = "deploy_stale";
  } else if (closed === 0) {
    economic_read_reason = "challenger_closed_zero";
  } else if (closed < THRESHOLDS.minimumClosedForReadableEconomics) {
    economic_read_reason = "sample_too_small";
  } else {
    economic_read_valid = true;
    economic_read_reason = "ok";
  }

  const avgMult = def(diag?.avgCapitalMultiplierOpened, 1);
  const earlyCount = def(diag?.earlyThesisFailureExitCount, 0);
  const defenseActivation: DefenseActivation = {
    structuralGateByPair: {
      activated: metrics.rejectedByStructuralPairCount > 0,
      rejections: metrics.rejectedByStructuralPairCount,
      note:
        metrics.rejectedByStructuralPairCount > 0
          ? `Gate rejeitou ${metrics.rejectedByStructuralPairCount} opps fora do conjunto de pares.`
          : "Nenhuma rejeição por pair — todas opps passaram no gate.",
    },
    capfloor: {
      activated: metrics.rejectedByCapfloorCount > 0,
      rejections: metrics.rejectedByCapfloorCount,
      note:
        metrics.rejectedByCapfloorCount > 0
          ? `Capfloor rejeitou ${metrics.rejectedByCapfloorCount} opps.`
          : "Nenhuma rejeição — capfloor não mordeu.",
    },
    degratio: {
      activated: metrics.rejectedByDegRatioCount > 0,
      rejections: metrics.rejectedByDegRatioCount,
      note:
        metrics.rejectedByDegRatioCount > 0
          ? `Degratio rejeitou ${metrics.rejectedByDegRatioCount} opps.`
          : "Nenhuma rejeição — degratio não mordeu.",
    },
    capitalMultiplier: {
      activated: avgMult < 1.0,
      avgMultiplier: avgMult,
      note:
        avgMult < 1.0
          ? `Multiplicador médio ${avgMult.toFixed(2)} — reduziu capital em trades abertos.`
          : "Multiplicador = 1.0 em todos — sizing adaptativo não reduziu capital.",
    },
    earlyThesisFailure: {
      activated: earlyCount > 0,
      exitCount: earlyCount,
      note:
        earlyCount > 0
          ? `${earlyCount} saída(s) por early thesis failure.`
          : "Nenhuma saída early — defesa não acionada.",
    },
  };

  const snapshot: Snapshot = {
    timestamp: new Date().toISOString(),
    url,
    economic_read_valid,
    economic_read_reason,
    structuralRiskManagedDiagnostics: diag ?? null,
    structuralRiskManagedComparison_vs_shadow_1000: compVs1000,
    structuralRiskManagedComparison_vs_exitrefine: compVsExitrefine,
    structuralExitKillDiagnostics: exitKillDiag ?? null,
    structuralExitKillComparison: exitKillComp ?? null,
    structuralExitKillWindow180Diagnostics: exitKillWindow180Diag ?? null,
    structuralExitKillWindow180Comparison: exitKillWindow180Comp ?? null,
    structuralExitKillWindow180CausalAudit: exitKillWindow180Causal ?? null,
    structuralLateExitDiagnostics: lateExitDiag ?? null,
    structuralLateExitComparison: lateExitComp ?? null,
    profileSummary,
    exitKillProfileSummary,
    exitKillWindow180ProfileSummary,
    lateExitProfileSummary,
    metrics,
    defenseActivation,
  };

  const outDir = path.join(__dirname, "..", "reports");
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  const outPath = path.join(outDir, "runtime_snapshot_latest.json");
  fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2), "utf-8");
  process.stdout.write(`Snapshot saved: ${outPath}\n`);
}

run().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
});
