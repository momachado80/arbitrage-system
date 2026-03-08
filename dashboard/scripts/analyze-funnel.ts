#!/usr/bin/env npx ts-node
/**
 * Analyze shadow pipeline funnel from /api/system/rejection-stats.
 * Run while dashboard is serving (npm run dev or production).
 * Usage: npx ts-node -P tsconfig.worker.json scripts/analyze-funnel.ts [BASE_URL]
 * Default: http://localhost:3000
 */

import * as https from "https";
import * as http from "http";

const BASE = process.argv[2] || "http://localhost:3000";

function fetchJson(url: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith("https") ? https : http;
    lib.get(url, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        try {
          resolve(JSON.parse(body));
        } catch {
          reject(new Error("Invalid JSON"));
        }
      });
    }).on("error", reject);
  });
}

async function main() {
  const url = `${BASE}/api/system/rejection-stats`;
  let data: Record<string, unknown>;
  try {
    data = (await fetchJson(url)) as Record<string, unknown>;
  } catch (e) {
    console.error("Failed to fetch", url, "\n", e instanceof Error ? e.message : e);
    process.exit(1);
  }
  const p = (data.pipeline || {}) as Record<string, unknown>;
  const totalD = Number(p.totalDispatches ?? 0);
  const totalE = Number(p.totalEvaluateCalls ?? 0);
  const totalX = Number(p.totalExecutionCalls ?? 0);
  const totalT = Number(p.totalShadowTradesOpened ?? 0);
  const early = (p.earlyExitCounts ?? {}) as Record<string, number>;

  console.log("\n=== SHADOW PIPELINE FUNNEL ===\n");
  console.log("1. FUNNEL COUNTS");
  console.log("   totalDispatches:        ", totalD);
  console.log("   totalEvaluateCalls:     ", totalE);
  console.log("   totalExecutionCalls:    ", totalX);
  console.log("   totalShadowTradesOpened:", totalT);

  console.log("\n2. EARLY EXIT COUNTS (by reason)");
  const entries = Object.entries(early).sort((a, b) => b[1] - a[1]);
  for (const [reason, count] of entries) {
    const pct = totalE > 0 ? ((count / totalE) * 100).toFixed(1) : "0";
    console.log(`   ${reason}: ${count} (${pct}% of evaluateCalls)`);
  }

  console.log("\n3. BOTTLENECK ANALYSIS");
  if (totalD === 0) {
    console.log("   → No dispatches yet. Ensure worker/bot or runCycle is running.");
  } else if (totalE === 0 && totalD > 0) {
    console.log("   → Dispatches occur but evaluate never called (NO_ENABLED_PROFILES or error).");
  } else {
    const top = entries[0];
    if (top) {
      const [reason, count] = top;
      const pct = totalE > 0 ? (count / totalE) * 100 : 0;
      console.log(`   → Dominant: ${reason} (${count} = ${pct.toFixed(1)}%)`);
    }
  }

  console.log("\n4. SYSTEM STATE");
  if (totalT > 0) console.log("   → Some shadow trades opened (C)");
  else if (totalX > 0 && totalT === 0) console.log("   → Reaching execution but rejecting most (B)");
  else if (totalE > 0 && totalX === 0) console.log("   → Not reaching execution (early exits before engine) (A)");
  else if (totalD > 0 && totalE === 0) console.log("   → Blocked by configuration/state (D)");
  else console.log("   → Insufficient data");

  console.log("\n5. RAW JSON (pipeline)\n");
  console.log(JSON.stringify({ pipeline: p }, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
