/**
 * Teste terminal de calibração de unidades reward→ciclo maker (6 mercados fixos).
 * Só recalibra o termo reward; resto do pilot inalterado.
 */

import { fetchGammaMarketRawJson } from "./clobMicrostructure";
import { resolveMarketRewardSignal } from "./polymarketRewards";
import { evaluatePilotMarket, pickPilotUniverse, waitForCatalogSnapshot } from "./rewardAwareMakerPilot";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function cyclesPerHourProxy(m: NormalizedMarket): number {
  const base = m.liquidity > 0 ? (m.volume / (m.liquidity + 600)) * 3.2 : 0.4;
  return r6(Math.min(14, Math.max(0.35, base)));
}

export type RewardUnitsCalibrationVerdict =
  | "calibrated_reward_still_not_viable"
  | "one_market_rescued_by_reward_units_calibration"
  | "multiple_markets_rescued_by_reward_units_calibration";

export interface RewardUnitsCalibrationMarketRow {
  marketId: string;
  marketTitle: string;
  verifiedTotalDailyRate: number;
  makerFillsPerDayAssumed: number;
  notionalAtRiskUsd: number;
  capitalDeployedUsd: number;
  poolAttributionShare: number;
  rewardUsdAttributedPerDay: number;
  calibratedRewardPerCycle: number;
  calibratedNetMakerCycle: number;
  supportingNote: string;
}

export interface RewardUnitsCalibrationPilotDigest {
  probeVersion: "reward-units-calibration-pilot-v1";
  readDisclaimer: string;
  marketsEvaluated: number;
  marketsWithPositiveNetCycleUnderCalibratedReward: number;
  rewardUnitsCalibrationVerdict: RewardUnitsCalibrationVerdict;
  rewardUnitsCalibrationSummaryLine: string;
  markets: RewardUnitsCalibrationMarketRow[];
  computedAt: string;
}

/**
 * Hipóteses explícitas (teste terminal, limite superior moderado):
 * - Cadência: round-trips maker/dia entre 3 e 18, ~1.8× cph (maker mais lento que taker).
 * - Notional em risco: 3% liquidez + piso/teto USD (proxy de tamanho de quote).
 * - Capital deployado: 4.5% liquidez + piso/teto (denominador alternativo documentado).
 * - Pool: fração do daily reward total atribuível a um pilot pequeno (teto 15%).
 * - Edge/ciclo: (USD atribuído/dia) / fillsPerDay / notionalAtRisk (yield por ciclo; cap 0.015).
 */
function calibratedRewardPerCycleTerminal(
  m: NormalizedMarket,
  verifiedDaily: number,
): {
  fillsPerDay: number;
  notionalAtRiskUsd: number;
  capitalDeployedUsd: number;
  poolAttributionShare: number;
  rewardUsdAttributedPerDay: number;
  calibratedRewardPerCycle: number;
  supportingNote: string;
} {
  const cph = cyclesPerHourProxy(m);
  const fillsPerDay = r6(Math.max(3, Math.min(18, cph * 1.8)));
  const notionalAtRiskUsd = r6(Math.max(900, Math.min(35_000, m.liquidity * 0.03 + 900)));
  const capitalDeployedUsd = r6(Math.max(1_500, Math.min(70_000, m.liquidity * 0.045 + 1_800)));
  const poolAttributionShare = r6(Math.min(0.15, 6_000 / (m.liquidity + 3_500)));
  const rewardUsdAttributedPerDay = r6(verifiedDaily * poolAttributionShare);
  const usdPerMakerCycle = rewardUsdAttributedPerDay / Math.max(1e-9, fillsPerDay);
  const calibratedRewardPerCycle = r6(Math.min(0.015, usdPerMakerCycle / Math.max(1e-6, notionalAtRiskUsd)));

  const supportingNote = [
    `cph=${cph}`,
    `fills_per_day=${fillsPerDay}`,
    `notional_at_risk_usd=${notionalAtRiskUsd}`,
    `capital_deployed_usd=${capitalDeployedUsd}`,
    `pool_share=${poolAttributionShare}`,
    `reward_usd_attr_day=${rewardUsdAttributedPerDay}`,
    `formula=((verified_daily*pool_share)/fills_per_day)/notional_at_risk`,
    `cap=0.015`,
  ].join("|");

  return {
    fillsPerDay,
    notionalAtRiskUsd,
    capitalDeployedUsd,
    poolAttributionShare,
    rewardUsdAttributedPerDay,
    calibratedRewardPerCycle,
    supportingNote,
  };
}

export async function buildRewardUnitsCalibrationPilotDigest(): Promise<RewardUnitsCalibrationPilotDigest> {
  await waitForCatalogSnapshot();
  const candidates = pickPilotUniverse();
  const markets: RewardUnitsCalibrationMarketRow[] = [];

  for (let i = 0; i < candidates.length; i++) {
    const m = candidates[i];
    const pilot = await evaluatePilotMarket(m);
    const raw = await fetchGammaMarketRawJson(m.id);
    const sig = await resolveMarketRewardSignal(raw);
    const verifiedDaily = sig.rewardSourceAvailable ? sig.verifiedTotalDailyRate : 0;

    const cal = calibratedRewardPerCycleTerminal(m, verifiedDaily);
    const calibratedNetMakerCycle = r6(
      pilot.estimatedNetMakerCycle - pilot.expectedRewardContribution + cal.calibratedRewardPerCycle,
    );

    markets.push({
      marketId: pilot.marketId,
      marketTitle: pilot.marketTitle,
      verifiedTotalDailyRate: verifiedDaily,
      makerFillsPerDayAssumed: cal.fillsPerDay,
      notionalAtRiskUsd: cal.notionalAtRiskUsd,
      capitalDeployedUsd: cal.capitalDeployedUsd,
      poolAttributionShare: cal.poolAttributionShare,
      rewardUsdAttributedPerDay: cal.rewardUsdAttributedPerDay,
      calibratedRewardPerCycle: cal.calibratedRewardPerCycle,
      calibratedNetMakerCycle,
      supportingNote: `${sig.detailNote}|${cal.supportingNote}`,
    });

    if (i < candidates.length - 1) {
      await new Promise(r => setTimeout(r, 100));
    }
  }

  const marketsWithPositiveNetCycleUnderCalibratedReward = markets.filter(m => m.calibratedNetMakerCycle > 0).length;
  const posCount = marketsWithPositiveNetCycleUnderCalibratedReward;

  let rewardUnitsCalibrationVerdict: RewardUnitsCalibrationVerdict;
  if (posCount >= 2) rewardUnitsCalibrationVerdict = "multiple_markets_rescued_by_reward_units_calibration";
  else if (posCount === 1) rewardUnitsCalibrationVerdict = "one_market_rescued_by_reward_units_calibration";
  else rewardUnitsCalibrationVerdict = "calibrated_reward_still_not_viable";

  const topCal = [...markets].sort((a, b) => b.calibratedNetMakerCycle - a.calibratedNetMakerCycle)[0]?.calibratedNetMakerCycle;
  const rewardUnitsCalibrationSummaryLine = `reward_units_calibration: verdict=${rewardUnitsCalibrationVerdict} | evaluated=${markets.length} pos_calibrated=${marketsWithPositiveNetCycleUnderCalibratedReward} | top_cal_net=${topCal ?? "n/a"}`;

  return {
    probeVersion: "reward-units-calibration-pilot-v1",
    readDisclaimer:
      "Teste terminal: mesmos 6 mercados; só recalibra reward/ciclo com cadência maker, notional, capital deployado e share do pool diário explícitos na nota. Limite superior moderado (pool_share≤15%, cap edge/ciclo 0.015). Não altera spread/adverse/inventário/hedge do pilot.",
    marketsEvaluated: markets.length,
    marketsWithPositiveNetCycleUnderCalibratedReward,
    rewardUnitsCalibrationVerdict,
    rewardUnitsCalibrationSummaryLine,
    markets,
    computedAt: new Date().toISOString(),
  };
}
