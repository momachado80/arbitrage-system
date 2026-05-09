/**
 * Refined Outcome Classification — desagrega "neutral" em subcategorias
 * para capturar tilts sutis que a classificação binária colapsa.
 * Não altera labels existentes; acrescenta leitura paralela.
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

export type RefinedOutcomeLabel =
  | "neutral_pure"
  | "neutral_with_small_positive_tilt"
  | "neutral_with_small_negative_tilt"
  | "weak_micro_positive"
  | "weak_micro_negative"
  | "recurring_micro_positive"
  | "recurring_micro_negative"
  | "excluded";

export type RefinedOverallVerdict =
  | "insufficient_sample"
  | "pure_neutral_dominant"
  | "tilted_positive"
  | "tilted_negative"
  | "mixed_no_clear_tilt"
  | "recurring_positive_signal"
  | "recurring_negative_signal";

export interface RefinedEpisodeRow {
  entryId: string;
  microBucketKey: string;
  originalLabel: string;
  refinedLabel: RefinedOutcomeLabel;
  spreadDelta: number | null;
  priceMeanDelta: number | null;
  netTiltScalar: number | null;
}

export interface RefinedOutcomeClassificationDigest {
  readDisclaimer: string;
  totalEligible: number;
  counts: Record<RefinedOutcomeLabel, number>;
  overallVerdict: RefinedOverallVerdict;
  positiveTiltRate: number | null;
  negativeTiltRate: number | null;
  pureNeutralRate: number | null;
  episodeDetails: RefinedEpisodeRow[];
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
}

function computePerMarket(
  before: MinimalPaperMarketLite[],
  after: MinimalPaperMarketLite[],
): { spreadDelta: number | null; priceMeanDelta: number | null; matched: number } {
  const afterMap = new Map(after.map(m => [m.id, m]));
  const spreadDeltas: number[] = [];
  const priceDeltas: number[] = [];
  for (const b of before) {
    const a = afterMap.get(b.id);
    if (!a) continue;
    spreadDeltas.push(r4(a.spread - b.spread));
    if (b.prices.length === a.prices.length) {
      for (let i = 0; i < b.prices.length; i++) {
        priceDeltas.push(r4(a.prices[i]! - b.prices[i]!));
      }
    }
  }
  if (spreadDeltas.length === 0) return { spreadDelta: null, priceMeanDelta: null, matched: 0 };
  return {
    spreadDelta: r4(spreadDeltas.reduce((a, b) => a + b, 0) / spreadDeltas.length),
    priceMeanDelta: priceDeltas.length > 0
      ? r4(priceDeltas.reduce((a, b) => a + b, 0) / priceDeltas.length)
      : null,
    matched: spreadDeltas.length,
  };
}

function classifyRefined(
  spreadDelta: number | null,
  priceMeanDelta: number | null,
  tiltPos: number,
  tiltNeg: number,
  weakPos: number,
  weakNeg: number,
): { label: RefinedOutcomeLabel; netTilt: number | null } {
  if (spreadDelta == null) return { label: "excluded", netTilt: null };
  const sd = spreadDelta;
  const pd = priceMeanDelta ?? 0;
  const netTilt = r4(-sd + pd);

  if (netTilt >= weakPos) return { label: "weak_micro_positive", netTilt };
  if (netTilt <= -weakNeg) return { label: "weak_micro_negative", netTilt };
  if (netTilt >= tiltPos) return { label: "neutral_with_small_positive_tilt", netTilt };
  if (netTilt <= -tiltNeg) return { label: "neutral_with_small_negative_tilt", netTilt };
  return { label: "neutral_pure", netTilt };
}

export function buildRefinedOutcomeClassification(
  entries: readonly MinimalPaperEntry[],
): RefinedOutcomeClassificationDigest {
  const tiltPos = envNum("REFINED_TILT_POS", 0.001);
  const tiltNeg = envNum("REFINED_TILT_NEG", 0.001);
  const weakPos = envNum("REFINED_WEAK_POS", 0.004);
  const weakNeg = envNum("REFINED_WEAK_NEG", 0.004);
  const minEpisodes = Math.max(1, Math.floor(envNum("REFINED_MIN_EPISODES", 4)));
  const recurringMinCount = Math.max(2, Math.floor(envNum("REFINED_RECURRING_MIN_COUNT", 3)));
  const recurringMinRate = envNum("REFINED_RECURRING_MIN_RATE", 0.35);

  const thresholdsUsed: Record<string, number> = {
    REFINED_TILT_POS: tiltPos,
    REFINED_TILT_NEG: tiltNeg,
    REFINED_WEAK_POS: weakPos,
    REFINED_WEAK_NEG: weakNeg,
    REFINED_MIN_EPISODES: minEpisodes,
    REFINED_RECURRING_MIN_COUNT: recurringMinCount,
    REFINED_RECURRING_MIN_RATE: recurringMinRate,
  };

  const closed = entries.filter(e => e.observedWindow);
  const rows: RefinedEpisodeRow[] = [];
  const counts: Record<RefinedOutcomeLabel, number> = {
    neutral_pure: 0,
    neutral_with_small_positive_tilt: 0,
    neutral_with_small_negative_tilt: 0,
    weak_micro_positive: 0,
    weak_micro_negative: 0,
    recurring_micro_positive: 0,
    recurring_micro_negative: 0,
    excluded: 0,
  };

  for (const e of closed) {
    const ow = e.observedWindow!;
    if (
      ow.observationalOutcomeLabel === "insufficient_data" ||
      ow.observationalOutcomeLabel === "component_missing_in_followup"
    ) {
      counts.excluded++;
      rows.push({
        entryId: e.id,
        microBucketKey: e.microBucketKey,
        originalLabel: ow.observationalOutcomeLabel,
        refinedLabel: "excluded",
        spreadDelta: null,
        priceMeanDelta: null,
        netTiltScalar: null,
      });
      continue;
    }

    const { spreadDelta, priceMeanDelta, matched } = computePerMarket(
      e.entrySnapshot.marketsLite, ow.marketsLiteAfter,
    );
    if (matched === 0) {
      counts.excluded++;
      rows.push({
        entryId: e.id, microBucketKey: e.microBucketKey,
        originalLabel: ow.observationalOutcomeLabel,
        refinedLabel: "excluded", spreadDelta: null, priceMeanDelta: null, netTiltScalar: null,
      });
      continue;
    }

    const { label, netTilt } = classifyRefined(spreadDelta, priceMeanDelta, tiltPos, tiltNeg, weakPos, weakNeg);
    counts[label]++;
    rows.push({
      entryId: e.id,
      microBucketKey: e.microBucketKey,
      originalLabel: ow.observationalOutcomeLabel,
      refinedLabel: label,
      spreadDelta,
      priceMeanDelta,
      netTiltScalar: netTilt,
    });
  }

  const eligible = rows.filter(r => r.refinedLabel !== "excluded").length;

  const positiveTiltCount = counts.neutral_with_small_positive_tilt + counts.weak_micro_positive + counts.recurring_micro_positive;
  const negativeTiltCount = counts.neutral_with_small_negative_tilt + counts.weak_micro_negative + counts.recurring_micro_negative;

  const byBucket = new Map<string, RefinedOutcomeLabel[]>();
  for (const r of rows) {
    if (r.refinedLabel === "excluded") continue;
    const arr = byBucket.get(r.microBucketKey) ?? [];
    arr.push(r.refinedLabel);
    byBucket.set(r.microBucketKey, arr);
  }
  for (const [bk, labels] of Array.from(byBucket.entries())) {
    const posLabels = labels.filter(l => l === "weak_micro_positive" || l === "neutral_with_small_positive_tilt");
    if (posLabels.length >= recurringMinCount && posLabels.length / labels.length >= recurringMinRate) {
      for (const r of rows) {
        if (r.microBucketKey === bk && (r.refinedLabel === "weak_micro_positive" || r.refinedLabel === "neutral_with_small_positive_tilt")) {
          counts[r.refinedLabel]--;
          r.refinedLabel = "recurring_micro_positive";
          counts.recurring_micro_positive++;
        }
      }
    }
    const negLabels = labels.filter(l => l === "weak_micro_negative" || l === "neutral_with_small_negative_tilt");
    if (negLabels.length >= recurringMinCount && negLabels.length / labels.length >= recurringMinRate) {
      for (const r of rows) {
        if (r.microBucketKey === bk && (r.refinedLabel === "weak_micro_negative" || r.refinedLabel === "neutral_with_small_negative_tilt")) {
          counts[r.refinedLabel]--;
          r.refinedLabel = "recurring_micro_negative";
          counts.recurring_micro_negative++;
        }
      }
    }
  }

  const positiveTiltRate = eligible > 0 ? r4(positiveTiltCount / eligible) : null;
  const negativeTiltRate = eligible > 0 ? r4(negativeTiltCount / eligible) : null;
  const pureNeutralRate = eligible > 0 ? r4(counts.neutral_pure / eligible) : null;

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let overallVerdict: RefinedOverallVerdict;
  if (eligible < minEpisodes) {
    overallVerdict = "insufficient_sample";
    blockingReasons.push(`Elegíveis ${eligible} < mínimo ${minEpisodes}.`);
  } else if (counts.recurring_micro_positive > 0 && counts.recurring_micro_negative === 0) {
    overallVerdict = "recurring_positive_signal";
    supportingReasons.push(
      `${counts.recurring_micro_positive} episódio(s) promovidos a recurring_micro_positive em ≥${recurringMinCount} ocorrências por bucket.`,
    );
  } else if (counts.recurring_micro_negative > 0 && counts.recurring_micro_positive === 0) {
    overallVerdict = "recurring_negative_signal";
    blockingReasons.push(`${counts.recurring_micro_negative} episódio(s) recurring_micro_negative.`);
  } else if (pureNeutralRate != null && pureNeutralRate >= 0.65) {
    overallVerdict = "pure_neutral_dominant";
    supportingReasons.push(
      `neutral_pure ${pureNeutralRate} ≥ 0.65: book genuinamente plano na maioria dos episódios.`,
    );
  } else if (positiveTiltCount > negativeTiltCount * 1.5 && positiveTiltCount >= 2) {
    overallVerdict = "tilted_positive";
    supportingReasons.push(
      `Tilt positivo: ${positiveTiltCount} episódios com tilt favorável vs ${negativeTiltCount} adverso.`,
    );
  } else if (negativeTiltCount > positiveTiltCount * 1.5 && negativeTiltCount >= 2) {
    overallVerdict = "tilted_negative";
    blockingReasons.push(`Tilt negativo dominante: ${negativeTiltCount} vs ${positiveTiltCount} positivo.`);
  } else {
    overallVerdict = "mixed_no_clear_tilt";
    supportingReasons.push("Distribuição mista sem tilt claro.");
  }

  supportingReasons.push(
    "netTiltScalar = −meanΔspread + meanΔprice; positivo = condições melhores, negativo = piores. Limiares por env.",
  );

  return {
    readDisclaimer:
      "Classificação refinada sobre a mesma base de dados do microEdgeAssessment. neutral_pure ≠ ausência de mercado; neutral_with_small_positive_tilt ≠ lucro; recurring ≠ edge validado.",
    totalEligible: eligible,
    counts,
    overallVerdict,
    positiveTiltRate,
    negativeTiltRate,
    pureNeutralRate,
    episodeDetails: rows.slice(0, 40),
    supportingReasons,
    blockingReasons,
    thresholdsUsed,
  };
}
