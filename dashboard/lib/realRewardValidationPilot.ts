/**
 * Validação de recompensa real — mesmo universo de 6 mercados do reward-aware maker pilot.
 * Só substitui a contribuição de reward; spread, hedge, inventário e adverse inalterados.
 */

import { fetchGammaMarketRawJson } from "./clobMicrostructure";
import {
  resolveMarketRewardSignal,
  type MarketRewardSignal,
  type RewardSignalKind,
} from "./polymarketRewards";
import { evaluatePilotMarket, pickPilotUniverse, waitForCatalogSnapshot } from "./rewardAwareMakerPilot";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function cyclesPerHourProxy(m: NormalizedMarket): number {
  const base = m.liquidity > 0 ? (m.volume / (m.liquidity + 600)) * 3.2 : 0.4;
  return r6(Math.min(14, Math.max(0.35, base)));
}

/**
 * Converte taxa diária verificada (payload CLOB/Gamma) numa pequena fração por ciclo maker,
 * coerente com a ordem de grandeza do proxy (documentado na nota).
 */
function dailyVerifiedToCycleEdge(m: NormalizedMarket, dailySum: number): number {
  if (!(dailySum > 0)) return 0;
  const cph = cyclesPerHourProxy(m);
  const perCycleFromDaily = dailySum / Math.max(1e-9, 24 * cph);
  const denom = Math.max(5_500, m.liquidity * 0.028 + 6_200);
  return r6(Math.min(0.0055, perCycleFromDaily / denom));
}

export type RealRewardValidationVerdict =
  | "no_real_reward_source_available"
  | "real_reward_does_not_rescue_viability"
  | "one_market_rescued_by_real_reward"
  | "multiple_markets_rescued_by_real_reward";

export interface RealRewardMarketRow {
  marketId: string;
  marketTitle: string;
  proxyRewardContribution: number;
  realRewardContribution: number | null;
  deltaRewardContribution: number | null;
  estimatedNetMakerCycleUnderRealReward: number;
  supportingNote: string;
}

export interface StrongestMarketUnderRealReward {
  marketId: string;
  marketTitle: string;
  estimatedNetMakerCycleUnderRealReward: number;
  verifiedTotalDailyRate: number;
}

export interface RealRewardValidationPilotDigest {
  probeVersion: "real-reward-validation-pilot-v1";
  readDisclaimer: string;
  rewardSourceType: RewardSignalKind | "mixed";
  rewardSourceAvailable: boolean;
  rewardProxyVsRealComparison: string;
  marketsEvaluated: number;
  marketsWithPositiveNetCycleUnderRealReward: number;
  strongestMarketsUnderRealReward: StrongestMarketUnderRealReward[];
  realRewardValidationVerdict: RealRewardValidationVerdict;
  realRewardValidationSummaryLine: string;
  markets: RealRewardMarketRow[];
  computedAt: string;
}

function aggregateSourceType(signals: MarketRewardSignal[]): RewardSignalKind | "mixed" {
  const kinds = new Set(signals.filter(s => s.rewardSourceAvailable).map(s => s.rewardSourceType));
  if (kinds.size === 0) return "none";
  if (kinds.size === 1) return Array.from(kinds)[0] as RewardSignalKind;
  return "mixed";
}

export async function buildRealRewardValidationPilotDigest(): Promise<RealRewardValidationPilotDigest> {
  await waitForCatalogSnapshot();
  const candidates = pickPilotUniverse();
  const rows: RealRewardMarketRow[] = [];
  const signals: MarketRewardSignal[] = [];
  const pairs: { row: RealRewardMarketRow; sig: MarketRewardSignal }[] = [];

  for (let i = 0; i < candidates.length; i++) {
    const m = candidates[i];
    const pilot = await evaluatePilotMarket(m);
    const raw = await fetchGammaMarketRawJson(m.id);
    const sig = await resolveMarketRewardSignal(raw);
    signals.push(sig);

    const proxyR = pilot.expectedRewardContribution;
    const realR = sig.rewardSourceAvailable ? dailyVerifiedToCycleEdge(m, sig.verifiedTotalDailyRate) : null;
    const delta = realR != null ? r6(realR - proxyR) : null;
    const netReal = r6(pilot.estimatedNetMakerCycle - pilot.expectedRewardContribution + (realR ?? pilot.expectedRewardContribution));

    const supportingNote = [
      `pilot_net=${pilot.estimatedNetMakerCycle}`,
      `proxy_reward=${proxyR}`,
      realR != null ? `real_reward_cycle_edge=${realR}` : "real_reward=null",
      `verified_daily_sum=${sig.verifiedTotalDailyRate}`,
      sig.detailNote,
      `transform=daily/(24*cph)/(0.028*liq+6200)_cap_0.0055`,
    ].join("|");

    const row: RealRewardMarketRow = {
      marketId: pilot.marketId,
      marketTitle: pilot.marketTitle,
      proxyRewardContribution: proxyR,
      realRewardContribution: realR,
      deltaRewardContribution: delta,
      estimatedNetMakerCycleUnderRealReward: netReal,
      supportingNote,
    };
    rows.push(row);
    pairs.push({ row, sig });

    if (i < candidates.length - 1) {
      await new Promise(r => setTimeout(r, 100));
    }
  }

  const anyVerified = signals.some(s => s.rewardSourceAvailable && s.verifiedTotalDailyRate > 0);
  const rewardSourceAvailable = anyVerified;
  const rewardSourceType = aggregateSourceType(signals);

  const proxyMean =
    rows.length > 0 ? r6(rows.reduce((s, x) => s + x.proxyRewardContribution, 0) / rows.length) : 0;
  const realVals = rows.map(x => x.realRewardContribution).filter((x): x is number => x != null && Number.isFinite(x));
  const realMean = realVals.length > 0 ? r6(realVals.reduce((a, b) => a + b, 0) / realVals.length) : 0;
  const deltaMean =
    realVals.length > 0
      ? r6(
          rows
            .map(x => x.deltaRewardContribution)
            .filter((x): x is number => x != null && Number.isFinite(x))
            .reduce((a, b) => a + b, 0) / realVals.length,
        )
      : 0;

  const rewardProxyVsRealComparison = `proxy_mean=${proxyMean}|real_mean_cycle_edge=${realMean}|delta_mean=${deltaMean}|markets_verified_daily_gt0=${signals.filter(s => s.verifiedTotalDailyRate > 0).length}/${rows.length}|aggregate_source=${rewardSourceType}`;

  const marketsWithPositiveNetCycleUnderRealReward = rows.filter(r => r.estimatedNetMakerCycleUnderRealReward > 0).length;

  const strongestMarketsUnderRealReward: StrongestMarketUnderRealReward[] = [...pairs]
    .sort((a, b) => b.row.estimatedNetMakerCycleUnderRealReward - a.row.estimatedNetMakerCycleUnderRealReward)
    .slice(0, 4)
    .map(({ row, sig }) => ({
      marketId: row.marketId,
      marketTitle: row.marketTitle,
      estimatedNetMakerCycleUnderRealReward: row.estimatedNetMakerCycleUnderRealReward,
      verifiedTotalDailyRate: sig.verifiedTotalDailyRate,
    }));

  let realRewardValidationVerdict: RealRewardValidationVerdict;
  if (!rewardSourceAvailable) {
    realRewardValidationVerdict = "no_real_reward_source_available";
  } else if (marketsWithPositiveNetCycleUnderRealReward >= 2) {
    realRewardValidationVerdict = "multiple_markets_rescued_by_real_reward";
  } else if (marketsWithPositiveNetCycleUnderRealReward === 1) {
    realRewardValidationVerdict = "one_market_rescued_by_real_reward";
  } else {
    realRewardValidationVerdict = "real_reward_does_not_rescue_viability";
  }

  const realRewardValidationSummaryLine = `real_reward_validation: verdict=${realRewardValidationVerdict} | evaluated=${rows.length} pos_under_real=${marketsWithPositiveNetCycleUnderRealReward} source=${rewardSourceType} available=${rewardSourceAvailable} | ${rewardProxyVsRealComparison}`;

  return {
    probeVersion: "real-reward-validation-pilot-v1",
    readDisclaimer:
      "Recompensa verificável: GET CLOB /rewards/markets/{conditionId} quando conditionId Gamma existe; senão soma de rates em clobRewards no JSON Gamma. Transformação diária→edge/ciclo é hipótese explícita (ver supportingNote); não altera spread/hedge/inventário/adverse do pilot.",
    rewardSourceType,
    rewardSourceAvailable,
    rewardProxyVsRealComparison,
    marketsEvaluated: rows.length,
    marketsWithPositiveNetCycleUnderRealReward,
    strongestMarketsUnderRealReward,
    realRewardValidationVerdict,
    realRewardValidationSummaryLine,
    markets: rows,
    computedAt: new Date().toISOString(),
  };
}
