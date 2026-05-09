import { NextResponse } from "next/server";
import { getPaperAnalyticsData } from "@/lib/paperSimulationService";
import { buildFeeImpactAudit } from "@/lib/graphProvenanceQualityAudit";
import { DEFAULT_PAPER_FEE_BUFFER_PER_LEG } from "@/lib/paperRealizedPnlSemantics";

export const dynamic = "force-dynamic";

/** Evita falha de `Response.json` (ex.: BigInt, valores não finitos) em payloads grandes. */
function jsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

function jsonResponse200(body: unknown): NextResponse {
  return new NextResponse(JSON.stringify(body, jsonReplacer), {
    status: 200,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export async function GET() {
  const emptyFeeAudit = buildFeeImpactAudit([], DEFAULT_PAPER_FEE_BUFFER_PER_LEG);
  const z = {
    totalTrades: 0,
    closedTrades: 0,
    winRate: 0,
    avgReturn: 0,
    avgPnL: 0,
    totalPnL: 0,
    profitFactor: 0,
    maxDrawdown: 0,
    avgHoldingTimeMs: 0,
    avgNetEdgeAtEntry: 0,
    avgFilledCapital: 0,
    utilizationRate: 0,
    pnlByOpportunityType: {},
    pnlBySourceType: {},
    averageCapacityConfidence: 0,
    pnlGrossRealized: 0,
    estimatedRoundTripFees: 0,
    totalEstimatedFees: 0,
    estimatedNetPnl: 0,
    feeImpactAudit: emptyFeeAudit,
    topGainCauses: [],
    topLossCauses: [],
    paperTradesCount: 0,
    opportunitiesSeenToday: 0,
    opportunitiesExecutableToday: 0,
    countByExitReason: {},
    avgPnLByExitReason: {},
    avgHoldingTimeByExitReason: {},
    avgCapturedEdgeRatioByExitReason: {},
    avgExpectedRemainingEdgeValueAtExit: 0,
    avgDrawdownFromPeakAtExit: 0,
    graphProvenanceClosedTrades: {
      closedCountByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
      totalPnLByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
      totalGrossPnLByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
      totalNetPnLByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
      avgNetPnLByProvenance: {
        equivalent: null,
        subset: null,
        exclusive: null,
        complementary_strict: null,
        complementary_relaxed: null,
        cycle: null,
        unknown: null,
      },
      countNetNegativeByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
      countGrossPositiveNetNegativeByProvenance: {
        equivalent: 0,
        subset: 0,
        exclusive: 0,
        complementary_strict: 0,
        complementary_relaxed: 0,
        cycle: 0,
        unknown: 0,
      },
    },
  };

  try {
    const data = getPaperAnalyticsData();
    const payload = {
      analytics: data.analytics,
      equityCurve: data.equityCurve,
      dailyOpportunityMetrics: data.dailyOpportunityMetrics,
      timestamp: new Date().toISOString(),
    };
    try {
      return jsonResponse200(payload);
    } catch (serErr) {
      console.error("[API /paper/analytics] serialization_failed", serErr);
      return jsonResponse200({
        analytics: z,
        equityCurve: [],
        dailyOpportunityMetrics: {
          todayUtc: new Date().toISOString().slice(0, 10),
          today: null,
          byDay: {},
          opportunitiesSeenToday: 0,
          opportunitiesExecutableToday: 0,
        },
        timestamp: new Date().toISOString(),
      });
    }
  } catch (err) {
    console.error("[API /paper/analytics]", err);
    return jsonResponse200({
      analytics: z,
      equityCurve: [],
      dailyOpportunityMetrics: {
        todayUtc: new Date().toISOString().slice(0, 10),
        today: null,
        byDay: {},
        opportunitiesSeenToday: 0,
        opportunitiesExecutableToday: 0,
      },
      timestamp: new Date().toISOString(),
    });
  }
}
