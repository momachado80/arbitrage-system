#!/usr/bin/env npx ts-node -P tsconfig.worker.json
/**
 * Fetch shadow audit from production. Discovers URL automatically when possible.
 *
 * URL resolution order:
 * 1. AUDIT_URL or DASHBOARD_URL env
 * 2. dashboard/.production-url file (one line: base URL)
 * 3. Known candidates from repo (800bb, eca1e)
 *
 * Usage:
 *   AUDIT_URL=https://your-dashboard.railway.app npx ts-node -P tsconfig.worker.json scripts/get-production-audit.ts
 *   echo "https://your-dashboard.railway.app" > .production-url && npx ts-node -P tsconfig.worker.json scripts/get-production-audit.ts
 */

import * as fs from "fs";
import * as path from "path";

const CANDIDATES = [
  "https://web-production-800bb.up.railway.app",
  "https://web-production-eca1e.up.railway.app",
];

async function fetchOk(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(5000) });
    return r.ok;
  } catch {
    return false;
  }
}

async function getAudit(baseUrl: string): Promise<unknown> {
  const res = await fetch(`${baseUrl.replace(/\/$/, "")}/api/shadow/audit`, {
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function main() {
  let baseUrl =
    process.env.AUDIT_URL ||
    process.env.DASHBOARD_URL ||
    null;

  if (!baseUrl) {
    const urlFile = path.join(__dirname, "..", ".production-url");
    if (fs.existsSync(urlFile)) {
      baseUrl = fs.readFileSync(urlFile, "utf-8").trim();
    }
  }

  if (!baseUrl) {
    process.stderr.write("Trying known candidates...\n");
    for (const c of CANDIDATES) {
      const ok = await fetchOk(`${c}/api/shadow/audit`);
      if (ok) {
        baseUrl = c;
        process.stderr.write(`Found: ${c}\n`);
        break;
      }
    }
  }

  if (!baseUrl) {
    process.stderr.write(
      "\nCould not find production dashboard URL.\n" +
        "Create dashboard/.production-url with your dashboard URL (one line).\n" +
        "Or set AUDIT_URL env var.\n"
    );
    process.exit(1);
  }

  const audit = await getAudit(baseUrl);
  process.stdout.write(JSON.stringify(audit, null, 2));
}

main().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
});
