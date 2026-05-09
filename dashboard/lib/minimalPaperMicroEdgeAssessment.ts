/**
 * Leitura observacional e conservadora de microedge acumulativo sobre entradas paper já fechadas.
 * Não é PnL monetizado; não altera gates nem probes; só agrega dados persistidos em entries[].
 */

import type { MinimalPaperEntry, MinimalPaperMarketLite, MinimalPaperObservedWindow } from "./minimalPaperExecutionProbe";

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

export type MicroEdgeReadVerdict =
  | "insufficient_sample"
  | "mostly_neutral"
  | "weak_positive_microedge"
  | "promising_small_recurring_edge"
  | "unstable_or_negative";

export type MicroEpisodeBucket = "micro_positive" | "micro_neutral" | "micro_negative" | "excluded";

export interface MicroEdgeAssessmentDigest {
  readNature: "observational_conservative_non_monetized_proxy";
  readDisclaimer: string;
  totalObservedPaperEpisodes: number;
  totalClosedPaperEpisodes: number;
  eligibleClosedEpisodesForMicroRead: number;
  excludedClosedEpisodesCount: number;
  microPositiveEpisodes: number;
  microNeutralEpisodes: number;
  microNegativeEpisodes: number;
  microPositiveRate: number | null;
  microNegativeRate: number | null;
  averageObservedMicroDrift: number | null;
  medianObservedMicroDrift: number | null;
  cumulativeConservativeMicroEdge: number;
  cumulativeConservativeMicroEdgePerEpisode: number | null;
  repeatedNeutralPattern: boolean;
  repeatedSmallPositivePattern: boolean;
  repeatedSmallNegativePattern: boolean;
  microEdgeStabilityScore: number;
  concentrationByMicroBucket: Record<string, { count: number; share: number }>;
  enoughSampleForMicroEdgeRead: boolean;
  microEdgeReadVerdict: MicroEdgeReadVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number | string>;
}

function compareSnapshots(
  before: MinimalPaperMarketLite[],
  after: MinimalPaperMarketLite[],
): {
  spreadDeltaMean: number | null;
  priceMeanDelta: number | null;
  maxAbs: number | null;
  matchedMarkets: number;
} {
  const afterMap = new Map(after.map(m => [m.id, m]));
  const spreadDeltas: number[] = [];
  const priceDeltas: number[] = [];
  let maxAbs = 0;
  let anyPrice = false;
  for (const b of before) {
    const a = afterMap.get(b.id);
    if (!a) continue;
    spreadDeltas.push(r4(a.spread - b.spread));
    if (b.prices.length === a.prices.length && b.prices.length > 0) {
      for (let i = 0; i < b.prices.length; i++) {
        const d = r4(a.prices[i]! - b.prices[i]!);
        priceDeltas.push(d);
        maxAbs = Math.max(maxAbs, Math.abs(d));
        anyPrice = true;
      }
    }
  }
  if (spreadDeltas.length === 0) {
    return { spreadDeltaMean: null, priceMeanDelta: null, maxAbs: anyPrice ? r4(maxAbs) : null, matchedMarkets: 0 };
  }
  const spreadDeltaMean = r4(spreadDeltas.reduce((x, y) => x + y, 0) / spreadDeltas.length);
  const priceMeanDelta =
    priceDeltas.length > 0 ? r4(priceDeltas.reduce((x, y) => x + y, 0) / priceDeltas.length) : null;
  return {
    spreadDeltaMean,
    priceMeanDelta,
    maxAbs: anyPrice ? r4(maxAbs) : null,
    matchedMarkets: spreadDeltas.length,
  };
}

/** Escalar observacional: spread a apertar (delta negativo) contribui +; média de Δpreço contribui com o sinal (proxy, não PnL). */
function driftScalar(
  spreadDeltaMean: number | null,
  priceMeanDelta: number | null,
): number | null {
  if (spreadDeltaMean == null && priceMeanDelta == null) return null;
  const s = spreadDeltaMean ?? 0;
  const p = priceMeanDelta ?? 0;
  if (spreadDeltaMean == null) return r4(p);
  if (priceMeanDelta == null) return r4(-s);
  return r4(-s + p);
}

function classifyEpisode(args: {
  ow: MinimalPaperObservedWindow;
  before: MinimalPaperMarketLite[];
  spreadTighten: number;
  spreadWiden: number;
  priceUp: number;
  priceDown: number;
  maxAbsVol: number;
}): { bucket: MicroEpisodeBucket; drift: number | null; volatile: boolean } {
  const { spreadDeltaMean, priceMeanDelta, maxAbs, matchedMarkets } = compareSnapshots(
    args.before,
    args.ow.marketsLiteAfter,
  );
  if (
    args.ow.observationalOutcomeLabel === "insufficient_data" ||
    args.ow.observationalOutcomeLabel === "component_missing_in_followup" ||
    matchedMarkets === 0
  ) {
    return { bucket: "excluded", drift: null, volatile: false };
  }
  const volatile = maxAbs != null && maxAbs > args.maxAbsVol;
  if (volatile) {
    return { bucket: "micro_negative", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: true };
  }
  if (args.ow.observationalOutcomeLabel === "book_quotes_unchanged_within_eps") {
    return { bucket: "micro_neutral", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: false };
  }
  const posSpread = spreadDeltaMean != null && spreadDeltaMean <= -args.spreadTighten;
  const posPrice = priceMeanDelta != null && priceMeanDelta >= args.priceUp;
  const negSpread = spreadDeltaMean != null && spreadDeltaMean >= args.spreadWiden;
  const negPrice = priceMeanDelta != null && priceMeanDelta <= -args.priceDown;
  const posSig = posSpread || posPrice;
  const negSig = negSpread || negPrice;
  if (posSig && negSig) {
    return { bucket: "micro_neutral", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: false };
  }
  if (posSig) {
    return { bucket: "micro_positive", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: false };
  }
  if (negSig) {
    return { bucket: "micro_negative", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: false };
  }
  return { bucket: "micro_neutral", drift: driftScalar(spreadDeltaMean, priceMeanDelta), volatile: false };
}

function maxBucketShare(counts: Record<string, number>, total: number): number {
  if (total <= 0) return 0;
  let m = 0;
  for (const c of Object.values(counts)) {
    m = Math.max(m, c);
  }
  return r4(m / total);
}

export function buildMicroEdgeSummaryLine(a: MicroEdgeAssessmentDigest): string {
  const er = a.eligibleClosedEpisodesForMicroRead;
  const pr = a.microPositiveRate != null ? r4(a.microPositiveRate * 100) : null;
  const nr = a.microNegativeRate != null ? r4(a.microNegativeRate * 100) : null;
  return `microEdge: ${a.microEdgeReadVerdict} | eligible=${er} pos%=${pr ?? "n/a"} neg%=${nr ?? "n/a"} | ${a.readNature}`;
}

export function buildMicroEdgeAssessmentFromEntries(entries: MinimalPaperEntry[]): MicroEdgeAssessmentDigest {
  const spreadTighten = envNum("MINIMAL_PAPER_MICRO_SPREAD_TIGHTEN", 0.008);
  const spreadWiden = envNum("MINIMAL_PAPER_MICRO_SPREAD_WIDEN", 0.008);
  const priceUp = envNum("MINIMAL_PAPER_MICRO_PRICE_UP", 0.004);
  const priceDown = envNum("MINIMAL_PAPER_MICRO_PRICE_DOWN", 0.004);
  const maxAbsVol = envNum("MINIMAL_PAPER_MICRO_MAX_ABS_VOL", 0.12);
  const minEligible = Math.max(1, Math.floor(envNum("MINIMAL_PAPER_MICRO_MIN_CLOSED_ELIGIBLE", 6)));
  const minPromising = Math.max(minEligible, Math.floor(envNum("MINIMAL_PAPER_MICRO_MIN_CLOSED_PROMISING", 8)));
  const weakPosRate = envNum("MINIMAL_PAPER_MICRO_WEAK_POS_RATE", 0.28);
  const promPosRate = envNum("MINIMAL_PAPER_MICRO_PROM_POS_RATE", 0.42);
  const maxBucketShareThr = envNum("MINIMAL_PAPER_MICRO_MAX_BUCKET_SHARE", 0.62);
  const negBlockRate = envNum("MINIMAL_PAPER_MICRO_NEG_RATE_BLOCK", 0.38);
  const mostlyNeutralRate = envNum("MINIMAL_PAPER_MICRO_MOSTLY_NEUTRAL_RATE", 0.58);
  const stabilityPromising = envNum("MINIMAL_PAPER_MICRO_STABILITY_PROMISING", 0.32);
  const negDomRatio = envNum("MINIMAL_PAPER_MICRO_NEG_DOMINANCE_RATIO", 2);

  const thresholdsUsed: Record<string, number | string> = {
    MINIMAL_PAPER_MICRO_SPREAD_TIGHTEN: spreadTighten,
    MINIMAL_PAPER_MICRO_SPREAD_WIDEN: spreadWiden,
    MINIMAL_PAPER_MICRO_PRICE_UP: priceUp,
    MINIMAL_PAPER_MICRO_PRICE_DOWN: priceDown,
    MINIMAL_PAPER_MICRO_MAX_ABS_VOL: maxAbsVol,
    MINIMAL_PAPER_MICRO_MIN_CLOSED_ELIGIBLE: minEligible,
    MINIMAL_PAPER_MICRO_MIN_CLOSED_PROMISING: minPromising,
    MINIMAL_PAPER_MICRO_WEAK_POS_RATE: weakPosRate,
    MINIMAL_PAPER_MICRO_PROM_POS_RATE: promPosRate,
    MINIMAL_PAPER_MICRO_MAX_BUCKET_SHARE: maxBucketShareThr,
    MINIMAL_PAPER_MICRO_NEG_RATE_BLOCK: negBlockRate,
    MINIMAL_PAPER_MICRO_MOSTLY_NEUTRAL_RATE: mostlyNeutralRate,
    MINIMAL_PAPER_MICRO_STABILITY_PROMISING: stabilityPromising,
    MINIMAL_PAPER_MICRO_NEG_DOMINANCE_RATIO: negDomRatio,
  };

  const totalObservedPaperEpisodes = entries.length;
  const closed = entries.filter(e => e.observedWindow);
  const totalClosedPaperEpisodes = closed.length;

  let microPositiveEpisodes = 0;
  let microNeutralEpisodes = 0;
  let microNegativeEpisodes = 0;
  let excludedClosedEpisodesCount = 0;
  const drifts: number[] = [];
  const bucketCounts: Record<string, number> = {};
  let volatileNegativeDrivers = 0;

  for (const e of closed) {
    const ow = e.observedWindow!;
    const { bucket, drift, volatile } = classifyEpisode({
      ow,
      before: e.entrySnapshot.marketsLite,
      spreadTighten,
      spreadWiden,
      priceUp,
      priceDown,
      maxAbsVol,
    });
    if (bucket === "excluded") {
      excludedClosedEpisodesCount++;
      continue;
    }
    if (drift != null) drifts.push(drift);
    const bk = e.microBucketKey;
    bucketCounts[bk] = (bucketCounts[bk] ?? 0) + 1;
    if (bucket === "micro_positive") microPositiveEpisodes++;
    else if (bucket === "micro_neutral") microNeutralEpisodes++;
    else {
      microNegativeEpisodes++;
      if (volatile) volatileNegativeDrivers++;
    }
  }

  const eligible = microPositiveEpisodes + microNeutralEpisodes + microNegativeEpisodes;
  const microPositiveRate = eligible > 0 ? r4(microPositiveEpisodes / eligible) : null;
  const microNegativeRate = eligible > 0 ? r4(microNegativeEpisodes / eligible) : null;
  const microNeutralRate = eligible > 0 ? r4(microNeutralEpisodes / eligible) : null;

  let cumulativeConservativeMicroEdge = 0;
  for (const e of closed) {
    const ow = e.observedWindow!;
    const { bucket } = classifyEpisode({
      ow,
      before: e.entrySnapshot.marketsLite,
      spreadTighten,
      spreadWiden,
      priceUp,
      priceDown,
      maxAbsVol,
    });
    if (bucket === "micro_positive") cumulativeConservativeMicroEdge += 1;
    else if (bucket === "micro_negative") cumulativeConservativeMicroEdge -= 1;
  }

  const cumulativeConservativeMicroEdgePerEpisode =
    eligible > 0 ? r4(cumulativeConservativeMicroEdge / eligible) : null;

  const averageObservedMicroDrift =
    drifts.length > 0 ? r4(drifts.reduce((a, b) => a + b, 0) / drifts.length) : null;
  const medianObservedMicroDrift = median(drifts);

  const maxShare = maxBucketShare(bucketCounts, eligible);
  const concentrationByMicroBucket: Record<string, { count: number; share: number }> = {};
  const keys = Object.keys(bucketCounts).sort((a, b) => bucketCounts[b]! - bucketCounts[a]!);
  for (const k of keys.slice(0, 15)) {
    const c = bucketCounts[k]!;
    concentrationByMicroBucket[k.slice(0, 120)] = {
      count: c,
      share: eligible > 0 ? r4(c / eligible) : 0,
    };
  }

  const neuR = microNeutralRate ?? 0;
  const posR = microPositiveRate ?? 0;
  const negR = microNegativeRate ?? 0;
  const microEdgeStabilityScore = r4(
    Math.max(
      0,
      Math.min(
        1,
        (1 - maxShare) * (1 - negR * 1.25) * (0.35 + Math.min(posR, neuR) * 0.65),
      ),
    ),
  );

  const repeatedNeutralPattern = eligible >= 4 && neuR >= mostlyNeutralRate && microNeutralEpisodes >= 3;
  const repeatedSmallPositivePattern = eligible >= 4 && posR >= weakPosRate && microPositiveEpisodes >= 3;
  const repeatedSmallNegativePattern = eligible >= 4 && negR >= weakPosRate && microNegativeEpisodes >= 3;

  const enoughSampleForMicroEdgeRead = eligible >= minEligible;

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let microEdgeReadVerdict: MicroEdgeReadVerdict = "mostly_neutral";

  if (!enoughSampleForMicroEdgeRead) {
    microEdgeReadVerdict = "insufficient_sample";
    blockingReasons.push(
      `Amostra elegível ${eligible} < mínimo ${minEligible} (episódios fechados com snapshot comparável e label utilizável).`,
    );
  } else if (
    negR > negBlockRate ||
    microNegativeEpisodes > microPositiveEpisodes * negDomRatio ||
    (negR >= weakPosRate && microEdgeStabilityScore < 0.12)
  ) {
    microEdgeReadVerdict = "unstable_or_negative";
    blockingReasons.push(
      `Sinal micro negativo ou instável dominante: negRate≈${negR}, stability≈${microEdgeStabilityScore}.`,
    );
    if (volatileNegativeDrivers > 0) {
      blockingReasons.push(
        `${volatileNegativeDrivers} episódio(s) com maxAbsPriceDelta > ${maxAbsVol} (livro volátil — tratado como micro_negative conservador).`,
      );
    }
  } else if (
    eligible >= minPromising &&
    posR >= promPosRate &&
    negR <= negBlockRate * 0.65 &&
    maxShare <= maxBucketShareThr &&
    microEdgeStabilityScore >= stabilityPromising
  ) {
    microEdgeReadVerdict = "promising_small_recurring_edge";
    supportingReasons.push(
      `Taxa micro-positiva ${posR} ≥ ${promPosRate}, neg ${negR} contida, concentração máx. bucket ${maxShare} ≤ ${maxBucketShareThr}, stability ${microEdgeStabilityScore} ≥ ${stabilityPromising}.`,
    );
  } else if (posR >= weakPosRate && negR <= negBlockRate * 0.92) {
    microEdgeReadVerdict = "weak_positive_microedge";
    supportingReasons.push(
      `Micro-positivos recorrentes (${posR} ≥ ${weakPosRate}) sem dominância negativa forte — leitura fraca, não conclusiva de edge monetizado.`,
    );
  } else if (neuR >= mostlyNeutralRate) {
    microEdgeReadVerdict = "mostly_neutral";
    supportingReasons.push(
      `Maioria neutra (${neuR} ≥ ${mostlyNeutralRate}): quotes estáveis dentro de eps ou micro-drift abaixo dos limiares direcionais — ausência de sinal positivo claro.`,
    );
  } else {
    microEdgeReadVerdict = "mostly_neutral";
    supportingReasons.push("Distribuição mista sem critérios para weak_positive ou promising; não forçar conclusão.");
  }

  supportingReasons.push(
    "Métricas derivadas de entrySnapshot vs observedWindow.marketsLiteAfter (spreads e preços); maxAbs elevado penaliza como instável.",
  );
  supportingReasons.push(
    "DriftScalar = −meanΔspread + meanΔprice (proxy observacional; não PnL nem sizing).",
  );

  const readDisclaimer =
    "Esta secção é observacional, conservadora e um proxy não monetizado de microestrutura de quotes; neutral ≠ lucro; micro_positive ≠ estratégia rentável até prova separada.";

  return {
    readNature: "observational_conservative_non_monetized_proxy",
    readDisclaimer,
    totalObservedPaperEpisodes,
    totalClosedPaperEpisodes,
    eligibleClosedEpisodesForMicroRead: eligible,
    excludedClosedEpisodesCount,
    microPositiveEpisodes,
    microNeutralEpisodes,
    microNegativeEpisodes,
    microPositiveRate,
    microNegativeRate,
    averageObservedMicroDrift,
    medianObservedMicroDrift,
    cumulativeConservativeMicroEdge,
    cumulativeConservativeMicroEdgePerEpisode,
    repeatedNeutralPattern,
    repeatedSmallPositivePattern,
    repeatedSmallNegativePattern,
    microEdgeStabilityScore,
    concentrationByMicroBucket,
    enoughSampleForMicroEdgeRead,
    microEdgeReadVerdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed,
  };
}
