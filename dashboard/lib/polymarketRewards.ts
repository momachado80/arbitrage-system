/**
 * Sinais de recompensa verificáveis (CLOB rewards API + fallback Gamma clobRewards).
 * Apenas leitura HTTP público; sem trading.
 */

const CLOB_BASE = (process.env.POLYMARKET_CLOB_HOST || "https://clob.polymarket.com").replace(/\/$/, "");
const FETCH_TIMEOUT_MS = 10_000;

export type RewardSignalKind = "clob_rewards_markets_endpoint" | "gamma_clob_rewards_json" | "none";

export interface MarketRewardSignal {
  rewardSourceType: RewardSignalKind;
  rewardSourceAvailable: boolean;
  /** Soma de taxas diárias reportadas pela API / JSON (unidades do payload; tipicamente USDC/dia agregado). */
  verifiedTotalDailyRate: number;
  detailNote: string;
}

function collectPositiveRates(o: unknown, sink: number[], depth = 0): void {
  if (depth > 24 || o == null) return;
  if (Array.isArray(o)) {
    for (const x of o) collectPositiveRates(x, sink, depth + 1);
    return;
  }
  if (typeof o !== "object") return;
  const r = o as Record<string, unknown>;
  const keys = ["rate_per_day", "ratePerDay", "rewardsDailyRate", "dailyRate", "native_daily_rate", "total_daily_rate"];
  for (const k of keys) {
    const v = r[k];
    if (typeof v === "number" && Number.isFinite(v) && v > 0) sink.push(v);
  }
  for (const k of Object.keys(r)) collectPositiveRates(r[k], sink, depth + 1);
}

function sumRates(values: number[]): number {
  const s = values.reduce((a, b) => a + b, 0);
  return Math.round(s * 1e9) / 1e9;
}

export function extractConditionId(raw: Record<string, unknown> | null): string | null {
  if (!raw) return null;
  const c = raw.conditionId ?? raw.condition_id;
  if (typeof c === "string" && c.startsWith("0x") && c.length >= 10) return c;
  return null;
}

function parseClobRewardsField(raw: Record<string, unknown>): unknown {
  const cr = raw.clobRewards;
  if (cr == null) return null;
  if (typeof cr === "string") {
    try {
      return JSON.parse(cr) as unknown;
    } catch {
      return null;
    }
  }
  return cr;
}

export function extractRewardSignalFromGamma(raw: Record<string, unknown> | null): MarketRewardSignal {
  if (!raw) {
    return {
      rewardSourceType: "none",
      rewardSourceAvailable: false,
      verifiedTotalDailyRate: 0,
      detailNote: "no_gamma_raw",
    };
  }
  const parsed = parseClobRewardsField(raw);
  const rates: number[] = [];
  collectPositiveRates(parsed, rates);
  const sum = sumRates(rates);
  if (sum <= 0) {
    return {
      rewardSourceType: "gamma_clob_rewards_json",
      rewardSourceAvailable: false,
      verifiedTotalDailyRate: 0,
      detailNote: "gamma_clob_rewards_missing_or_zero_rates",
    };
  }
  return {
    rewardSourceType: "gamma_clob_rewards_json",
    rewardSourceAvailable: true,
    verifiedTotalDailyRate: sum,
    detailNote: `gamma_clob_rewards_sum_daily=${sum}`,
  };
}

export async function fetchClobRewardSignalByCondition(conditionId: string): Promise<MarketRewardSignal> {
  if (!conditionId) {
    return {
      rewardSourceType: "none",
      rewardSourceAvailable: false,
      verifiedTotalDailyRate: 0,
      detailNote: "missing_condition_id",
    };
  }
  try {
    const url = `${CLOB_BASE}/rewards/markets/${encodeURIComponent(conditionId)}?limit=80`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      return {
        rewardSourceType: "clob_rewards_markets_endpoint",
        rewardSourceAvailable: false,
        verifiedTotalDailyRate: 0,
        detailNote: `clob_rewards_http_${res.status}`,
      };
    }
    const j = (await res.json()) as unknown;
    const rates: number[] = [];
    collectPositiveRates(j, rates);
    const sum = sumRates(rates);
    if (sum <= 0) {
      return {
        rewardSourceType: "clob_rewards_markets_endpoint",
        rewardSourceAvailable: false,
        verifiedTotalDailyRate: 0,
        detailNote: "clob_rewards_payload_zero_sum",
      };
    }
    return {
      rewardSourceType: "clob_rewards_markets_endpoint",
      rewardSourceAvailable: true,
      verifiedTotalDailyRate: sum,
      detailNote: `clob_rewards_markets_endpoint_sum_daily=${sum}`,
    };
  } catch (e) {
    return {
      rewardSourceType: "clob_rewards_markets_endpoint",
      rewardSourceAvailable: false,
      verifiedTotalDailyRate: 0,
      detailNote: `clob_rewards_fetch_error:${e instanceof Error ? e.message : String(e)}`,
    };
  }
}

/** Preferência: CLOB por conditionId; se falhar, Gamma clobRewards. */
export async function resolveMarketRewardSignal(raw: Record<string, unknown> | null): Promise<MarketRewardSignal> {
  const cid = extractConditionId(raw);
  if (cid) {
    const clob = await fetchClobRewardSignalByCondition(cid);
    if (clob.rewardSourceAvailable) return clob;
  }
  return extractRewardSignalFromGamma(raw);
}
