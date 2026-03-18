#!/usr/bin/env node
/**
 * Deploy check: validates production /api/shadow/audit exposes structural risk managed feature.
 *
 * URL resolution: AUDIT_URL | DASHBOARD_URL | .production-url | known candidates
 *
 * Usage: npm run shadow:deploy-check  (from dashboard/)
 */

import * as fs from "fs";
import * as path from "path";
import { httpGet } from "./shadow-http";

const PRODUCTION_CANDIDATES = [
  "https://dashboard-next-production-b126.up.railway.app",
  "https://web-production-800bb.up.railway.app",
  "https://web-production-eca1e.up.railway.app",
];
const EXPECTED_PROFILE_ID = "shadow_1000_structural_riskmanaged_v1";
const FETCH_TIMEOUT_MS = 15000;

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
      "No production URL. Set AUDIT_URL, create dashboard/.production-url, or ensure candidates are reachable."
    );
  }
  return base.replace(/\/$/, "");
}

interface DeployCheckResult {
  conclusion: "DEPLOY_OK" | "DEPLOY_STALE";
  checks: {
    structuralRiskManagedDiagnostics_exists: boolean;
    profileId_correct: boolean;
    structuralRiskManagedComparison_exists: boolean;
    profile_in_profileSummaries: boolean;
    profile_in_rejectionCountsByProfile: boolean;
  };
  url: string;
  timestamp: string;
}

async function main(): Promise<void> {
  const baseUrl = await getBaseUrl();
  const url = `${baseUrl}/api/shadow/audit`;

  let audit: Record<string, unknown>;
  try {
    const res = await httpGet(url, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    audit = (await res.json()) as Record<string, unknown>;
  } catch (e) {
    const result: DeployCheckResult = {
      conclusion: "DEPLOY_STALE",
      checks: {
        structuralRiskManagedDiagnostics_exists: false,
        profileId_correct: false,
        structuralRiskManagedComparison_exists: false,
        profile_in_profileSummaries: false,
        profile_in_rejectionCountsByProfile: false,
      },
      url,
      timestamp: new Date().toISOString(),
    };
    process.stdout.write(JSON.stringify(result, null, 2) + "\n");
    process.stderr.write(`Fetch failed: ${e instanceof Error ? e.message : String(e)}\n`);
    process.exit(1);
  }

  const diag = audit.structuralRiskManagedDiagnostics as Record<string, unknown> | undefined;
  const comp = audit.structuralRiskManagedComparison as Record<string, unknown> | undefined;
  const summaries = audit.profileSummaries as Array<{ profileId: string }> | undefined;
  const rejections = audit.rejectionCountsByProfile as Record<string, unknown> | undefined;

  const hasDiag = diag != null && typeof diag === "object";
  const profileIdOk = hasDiag && diag.profileId === EXPECTED_PROFILE_ID;
  const hasComp = comp != null && typeof comp === "object";
  const inSummaries =
    Array.isArray(summaries) &&
    summaries.some((p) => p?.profileId === EXPECTED_PROFILE_ID);
  const inRejections =
    rejections != null &&
    typeof rejections === "object" &&
    EXPECTED_PROFILE_ID in rejections;

  const allOk =
    hasDiag && profileIdOk && hasComp && inSummaries && inRejections;

  const result: DeployCheckResult = {
    conclusion: allOk ? "DEPLOY_OK" : "DEPLOY_STALE",
    checks: {
      structuralRiskManagedDiagnostics_exists: hasDiag,
      profileId_correct: profileIdOk,
      structuralRiskManagedComparison_exists: hasComp,
      profile_in_profileSummaries: inSummaries,
      profile_in_rejectionCountsByProfile: inRejections,
    },
    url,
    timestamp: new Date().toISOString(),
  };

  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
  process.exit(allOk ? 0 : 1);
}

main().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
});
