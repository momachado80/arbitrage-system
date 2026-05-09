/**
 * Capturability Assessment — proxy observacional e conservador que estima se existia
 * pequena captura possível entre entrySnapshot e observedWindow, mesmo com book aparentemente parado.
 * Não assume fill perfeito nem PnL monetizado real.
 */

import type {
  MinimalPaperEntry,
  MinimalPaperMarketLite,
} from "./minimalPaperExecutionProbe";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function median(nums: number[]): number | null {
  if (nums.length === 0) return null;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : r4((s[m - 1]! + s[m]!) / 2);
}

export type CapturabilityVerdict =
  | "insufficient_sample"
  | "no_capturable_signal"
  | "weak_capturable_signal"
  | "recurring_capturable_signal"
  | "adverse_capture_conditions";

export interface CapturabilityAssessmentDigest {
  readNature: "observational_conservative_capture_proxy";
  readDisclaimer: string;
  totalEligibleEpisodesForCapturability: number;
  capturableMicroPositiveEpisodes: number;
  capturableMicroNeutralEpisodes: number;
  capturableMicroNegativeEpisodes: number;
  capturablePositiveRate: number | null;
  averageCapturableProxy: number | null;
  medianCapturableProxy: number | null;
  cumulativeCapturableProxy: number;
  capturabilityVerdict: CapturabilityVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
}

function computeEpisodeCapturability(
  before: MinimalPaperMarketLite[],
  after: MinimalPaperMarketLite[],
  feeProxy: number,
  slippagePct: number,
): { proxy: number | null; matched: number } {
  const afterMap = new Map(after.map(m => [m.id, m]));
  const spreadImprovements: number[] = [];
  const priceAbsMoves: number[] = [];

  for (const b of before) {
    const a = afterMap.get(b.id);
    if (!a) continue;
    const si = Math.max(0, b.spread - a.spread);
    spreadImprovements.push(r4(si));
    if (b.prices.length === a.prices.length) {
      for (let i = 0; i < b.prices.length; i++) {
        priceAbsMoves.push(r4(Math.abs(a.prices[i]! - b.prices[i]!)));
      }
    }
  }

  if (spreadImprovements.length === 0) return { proxy: null, matched: 0 };

  const meanSI = spreadImprovements.reduce((a, b) => a + b, 0) / spreadImprovements.length;
  const meanPM = priceAbsMoves.length > 0
    ? priceAbsMoves.reduce((a, b) => a + b, 0) / priceAbsMoves.length
    : 0;

  const avgEntrySpread = before.reduce((s, m) => s + m.spread, 0) / before.length;
  const netPriceMoveBeyondSpread = Math.max(0, meanPM - avgEntrySpread * slippagePct);

  const proxy = r4(meanSI * 0.5 + netPriceMoveBeyondSpread * 0.4 - feeProxy);
  return { proxy, matched: spreadImprovements.length };
}

export function buildCapturabilityAssessment(
  entries: readonly MinimalPaperEntry[],
): CapturabilityAssessmentDigest {
  const feeProxy = envNum("CAPTURE_FEE_PROXY", 0.003);
  const slippagePct = envNum("CAPTURE_SLIPPAGE_PCT", 0.5);
  const positiveThreshold = envNum("CAPTURE_POSITIVE_THRESHOLD", 0.001);
  const negativeThreshold = envNum("CAPTURE_NEGATIVE_THRESHOLD", 0.002);
  const minEpisodes = Math.max(1, Math.floor(envNum("CAPTURE_MIN_EPISODES", 4)));
  const minPosRateRecurring = envNum("CAPTURE_MIN_POS_RATE_RECURRING", 0.4);
  const maxNegRateBlock = envNum("CAPTURE_MAX_NEG_RATE_BLOCK", 0.35);

  const thresholdsUsed: Record<string, number> = {
    CAPTURE_FEE_PROXY: feeProxy,
    CAPTURE_SLIPPAGE_PCT: slippagePct,
    CAPTURE_POSITIVE_THRESHOLD: positiveThreshold,
    CAPTURE_NEGATIVE_THRESHOLD: negativeThreshold,
    CAPTURE_MIN_EPISODES: minEpisodes,
    CAPTURE_MIN_POS_RATE_RECURRING: minPosRateRecurring,
    CAPTURE_MAX_NEG_RATE_BLOCK: maxNegRateBlock,
  };

  const closed = entries.filter(e => e.observedWindow);
  const proxies: number[] = [];
  let pos = 0;
  let neu = 0;
  let neg = 0;

  for (const e of closed) {
    const ow = e.observedWindow!;
    if (
      ow.observationalOutcomeLabel === "insufficient_data" ||
      ow.observationalOutcomeLabel === "component_missing_in_followup"
    ) continue;
    const { proxy, matched } = computeEpisodeCapturability(
      e.entrySnapshot.marketsLite, ow.marketsLiteAfter, feeProxy, slippagePct,
    );
    if (proxy == null || matched === 0) continue;
    proxies.push(proxy);
    if (proxy > positiveThreshold) pos++;
    else if (proxy < -negativeThreshold) neg++;
    else neu++;
  }

  const eligible = proxies.length;
  const posRate = eligible > 0 ? r4(pos / eligible) : null;
  const avg = eligible > 0 ? r4(proxies.reduce((a, b) => a + b, 0) / eligible) : null;
  const med = median(proxies);
  const cumulative = r4(proxies.reduce((a, b) => a + b, 0));

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let verdict: CapturabilityVerdict;
  if (eligible < minEpisodes) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Episódios elegíveis ${eligible} < mínimo ${minEpisodes}.`);
  } else {
    const negRate = eligible > 0 ? neg / eligible : 0;
    if (negRate >= maxNegRateBlock) {
      verdict = "adverse_capture_conditions";
      blockingReasons.push(
        `Taxa de captura adversa (${r4(negRate)}) ≥ ${maxNegRateBlock}: condições de captura desfavoráveis observadas.`,
      );
    } else if (posRate != null && posRate >= minPosRateRecurring && pos >= 3) {
      verdict = "recurring_capturable_signal";
      supportingReasons.push(
        `Taxa capturable-positiva ${posRate} ≥ ${minPosRateRecurring} com ${pos} episódios — sinal recorrente (proxy, não PnL real).`,
      );
    } else if (pos > 0) {
      verdict = "weak_capturable_signal";
      supportingReasons.push(
        `${pos} episódio(s) com proxy>0 (posRate=${posRate}); fraco mas existente.`,
      );
    } else {
      verdict = "no_capturable_signal";
      supportingReasons.push(
        "Nenhum episódio ultrapassou positiveThreshold; book parado sem spread improvement significativo.)",
      );
    }
  }

  supportingReasons.push(
    "Proxy = 0.5·meanSpreadImprovement + 0.4·max(0, meanPriceMove − entrySpread·slippage%) − feeProxy. Positivo sugere condições de captura favoráveis, não lucro.",
  );

  return {
    readNature: "observational_conservative_capture_proxy",
    readDisclaimer:
      "Proxy de capturabilidade — não é PnL real; não assume fill perfeito; não assume direção da posição. Spread improvement e price movement são condições necessárias, não suficientes.",
    totalEligibleEpisodesForCapturability: eligible,
    capturableMicroPositiveEpisodes: pos,
    capturableMicroNeutralEpisodes: neu,
    capturableMicroNegativeEpisodes: neg,
    capturablePositiveRate: posRate,
    averageCapturableProxy: avg,
    medianCapturableProxy: med,
    cumulativeCapturableProxy: cumulative,
    capturabilityVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed,
  };
}
