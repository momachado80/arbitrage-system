/**
 * Fetch shadow audit from dashboard API and output formatted summary.
 * Usage: AUDIT_URL=https://your-dashboard.railway.app npx ts-node scripts/fetch-shadow-audit.ts
 * Or:   npx ts-node scripts/fetch-shadow-audit.ts  (defaults to http://localhost:3000)
 */

const AUDIT_URL = process.env.AUDIT_URL || "http://localhost:3000";

async function main() {
  const url = `${AUDIT_URL.replace(/\/$/, "")}/api/shadow/audit`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  const audit = (await res.json()) as {
    timestamp: string;
    negativeExpectancy: boolean;
    profileSummaries: Array<{
      profileId: string;
      totalClosed: number;
      avgRealizedPnL: number;
      medianRealizedPnL: number;
      winRate: number;
      lossRate: number;
      avgHoldingTimeMs: number;
      avgFilledCapital: number;
      avgObservedEdgeAtEntry: number;
      avgCapturableEdgeAtEntry: number;
      totalRealizedPnL: number;
      sumWins: number;
      sumLosses: number;
    }>;
    byProfile: Record<string, { byOpportunityType: Record<string, { count: number; totalPnL: number; avgPnL: number }>; byExitReason: Record<string, { count: number; totalPnL: number; avgPnL: number }>; byHoldingBucket: Record<string, { count: number; totalPnL: number; avgPnL: number }> }>;
    worst20: Array<Record<string, unknown>>;
    best20: Array<Record<string, unknown>>;
    safestNextChange: string;
  };

  console.log("\n=== SHADOW CLOSED TRADE AUDIT ===\n");
  console.log("Timestamp:", audit.timestamp);
  console.log("negativeExpectancy:", audit.negativeExpectancy);

  console.log("\n--- profileSummaries ---");
  for (const p of audit.profileSummaries) {
    console.log(JSON.stringify(p, null, 2));
  }

  console.log("\n--- byProfile.byOpportunityType ---");
  for (const [prof, data] of Object.entries(audit.byProfile || {})) {
    console.log(prof + ":", JSON.stringify(data.byOpportunityType, null, 2));
  }

  console.log("\n--- byProfile.byExitReason ---");
  for (const [prof, data] of Object.entries(audit.byProfile || {})) {
    console.log(prof + ":", JSON.stringify(data.byExitReason, null, 2));
  }

  console.log("\n--- byProfile.byHoldingBucket ---");
  for (const [prof, data] of Object.entries(audit.byProfile || {})) {
    console.log(prof + ":", JSON.stringify(data.byHoldingBucket, null, 2));
  }

  console.log("\n--- worst20 (first 5) ---");
  audit.worst20.slice(0, 5).forEach((t, i) => console.log(i + 1, JSON.stringify(t)));

  console.log("\n--- best20 (first 5) ---");
  audit.best20.slice(0, 5).forEach((t, i) => console.log(i + 1, JSON.stringify(t)));

  console.log("\n--- safestNextChange ---");
  console.log(audit.safestNextChange);
  console.log("\n");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
