/**
 * Classificação estrutural PURA para micro-lanes derivadas de violações `equivalence`.
 * O builder pode rotular como "equivalent" por Jaccard alto sem identidade de evento;
 * aqui re-particionamos por hipótese estrutural mínima (partição competitiva, escada
 * monótona, equivalência lexical forte, resto).
 */

import type { NormalizedMarket } from "./polymarketClient";
import { tokenize, jaccardSimilarity } from "./marketRelationBuilder";

export type EquivalenceMicroStructuralLane =
  | "graph_equivalence_micro"
  | "graph_subset_micro"
  | "graph_exclusive_micro";

/** Auditar por que o par foi colocado em cada bucket. */
export type StructuralAssignmentReason =
  | "competitive_partition_balance_of_power"
  | "competitive_partition_rival_sports_entities"
  | "competitive_partition_partisan_chamber_or_grid"
  | "monotonic_threshold_or_date_ladder"
  | "true_equivalence_identical_informative_tokens"
  | "residual_not_pure_equivalence_nor_monotonic_subset";

const NUMERIC_RUN_RE = /\d[\d,]*(?:\.\d+)?[kmb]?/gi;

function numericQuestionTemplate(q: string): string {
  return q
    .toLowerCase()
    .replace(NUMERIC_RUN_RE, "#")
    .replace(/[^a-z0-9#]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Tokens informativos + marcadores estruturais que o tokenize base remove (ex.: D/R de uma letra). */
function enrichedStructuralTokens(q: string): Set<string> {
  const s = new Set(tokenize(q));
  const lower = q.toLowerCase();
  if (/\bbalance\b\s+of\s+power\b|\bpower\s+balance\b/i.test(lower)) s.add("__struct_bop");
  if (/\bdemocratic\b|\bdemocrat\b|\bd\s+senate|\bd\s+house|\(d\)/i.test(q)) s.add("__struct_dem");
  if (/\brepublican\b|\br\s+senate|\br\s+house|\(r\)|\bgop\b/i.test(q)) s.add("__struct_rep");
  if (/\bsenate\b/i.test(lower)) s.add("__struct_senate");
  if (/\bhouse\b/i.test(lower)) {
    const polCtx =
      /\bsenate\b|\bdemocrat\b|\brepublican\b|\bd\s+senate|\bd\s+house|\br\s+senate|\br\s+house|\bgop\b|\bcongress\b/i.test(
        lower
      );
    if (polCtx) s.add("__struct_house");
  }
  return s;
}

function normQ(q: string): string {
  return q.toLowerCase().replace(/\s+/g, " ").trim();
}

/** Normalização para equivalência lexical (variação superficial de pontuação/espacos). */
function normEquivalenceSurface(q: string): string {
  return q
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[ \t]*[?!.,;:]+[ \t]*$/g, "")
    .trim();
}

function hasBothChambers(e: Set<string>): boolean {
  return e.has("__struct_senate") && e.has("__struct_house");
}

/**
 * Mesmo multiconjunto de tokens estruturais (ex.: D+R+Senate+House) mas permutação de outcomes
 * (D Senate / R House vs R Senate / D House) — partição competitiva, não equivalência.
 */
function isCompetitivePartisanChamberPermutation(qa: string, qb: string): boolean {
  const ea = enrichedStructuralTokens(qa);
  const eb = enrichedStructuralTokens(qb);
  if (!setsEqual(ea, eb)) return false;
  if (!hasBothChambers(ea)) return false;
  if (!ea.has("__struct_dem") && !ea.has("__struct_rep")) return false;
  return normEquivalenceSurface(qa) !== normEquivalenceSurface(qb);
}

function setsEqual(a: Set<string>, b: Set<string>): boolean {
  if (a.size !== b.size) return false;
  for (const x of Array.from(a)) {
    if (!b.has(x)) return false;
  }
  return true;
}

/** Balance of Power e variantes de grelha partidária/câmaras. */
function isCompetitivePartitionBalanceOfPower(qa: string, qb: string): boolean {
  const a = qa.toLowerCase();
  const b = qb.toLowerCase();
  const bop = /\bbalance\b\s+of\s+power\b|\bpower\s+balance\b/i;
  return bop.test(a) && bop.test(b);
}

/** Paris FC vs PSG / rivalidade explícita em contexto desportivo (não mesma equipa / não P(A)=P(B)). */
function isCompetitiveRivalSportsEntities(qa: string, qb: string): boolean {
  const a = qa.toLowerCase();
  const b = qb.toLowerCase();
  const sportsCtx = /top\s*4|top\s+four|ligue|uefa|champions|premier|serie|finish|standings|relegation/i;
  if (!sportsCtx.test(a) || !sportsCtx.test(b)) return false;
  const hasParisA = /\bparis\s*fc\b|\bparis\b/.test(a) && !/\bparis\s*sg\b/.test(a);
  const hasPsgB = /\bpsg\b|\bparis\s*sg\b/.test(b);
  const hasParisB = /\bparis\s*fc\b|\bparis\b/.test(b) && !/\bparis\s*sg\b/.test(b);
  const hasPsgA = /\bpsg\b|\bparis\s*sg\b/.test(a);
  if ((hasParisA && hasPsgB) || (hasParisB && hasPsgA)) return true;
  return false;
}

/**
 * Grelha Senate+House com assinatura partidária distinta ou forte sobreposição lexical sem identidade de texto.
 * Exige ambas as câmaras em cada pergunta (alinhado a variantes BoP / controlo do Congresso).
 */
function isCompetitivePartisanChamberGrid(qa: string, qb: string): boolean {
  const ea = enrichedStructuralTokens(qa);
  const eb = enrichedStructuralTokens(qb);
  if (!hasBothChambers(ea) || !hasBothChambers(eb)) return false;
  if (normEquivalenceSurface(qa) === normEquivalenceSurface(qb)) return false;

  const demA = ea.has("__struct_dem");
  const repA = ea.has("__struct_rep");
  const demB = eb.has("__struct_dem");
  const repB = eb.has("__struct_rep");
  const sig = (d: boolean, r: boolean) => `${d ? "1" : "0"}${r ? "1" : "0"}`;
  if (sig(demA, repA) !== sig(demB, repB)) return true;
  if ((demA && repB) || (repA && demB)) return true;

  const jac = jaccardSimilarity(ea, eb);
  if (jac >= 0.55 && (demA || repA) && (demB || repB)) return true;
  return false;
}

/** Parse um número monetário / limiar a partir do texto (primeiro valor significativo). */
function extractPrimaryThresholdValue(q: string): number | null {
  const lower = q.toLowerCase();
  const re = /([\d,.]+)\s*([kmb])?(?=\s|$|[^\w])/i;
  const m = lower.match(re);
  if (!m) return null;
  const raw = m[1]!.replace(/,/g, "");
  const n = Number.parseFloat(raw);
  if (!Number.isFinite(n)) return null;
  const suf = (m[2] || "").toLowerCase();
  let mult = 1;
  if (suf === "k") mult = 1e3;
  else if (suf === "m") mult = 1e6;
  else if (suf === "b") mult = 1e9;
  return n * mult;
}

const MONTH_ORDER: Record<string, number> = {
  jan: 0,
  feb: 1,
  mar: 2,
  apr: 3,
  may: 4,
  jun: 5,
  jul: 6,
  aug: 7,
  sep: 8,
  oct: 9,
  nov: 10,
  dec: 11,
};

function extractFirstMonthOrdinal(q: string): number | null {
  const m = q.toLowerCase().match(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b/);
  if (!m) return null;
  const key = m[1]!.slice(0, 3) as keyof typeof MONTH_ORDER;
  const o = MONTH_ORDER[key];
  return typeof o === "number" ? o : null;
}

function stripMonthNames(q: string): string {
  return q
    .toLowerCase()
    .replace(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b/g, "#")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Mesmo molde após colapsar dígitos, palavras-chave de escada, e dois limiares distintos
 * (above X vs above Y, OpenSea FDV, top N, etc.) ou mesma frase com mês distinto (before June vs before July).
 */
function isMonotonicThresholdOrDateLadder(qa: string, qb: string): boolean {
  const kw = /above|at least|before|after|more than|less than|over|under|minimum|maximum|top\s*#|finish|fdv|market\s*cap/i;
  if (!kw.test(qa) || !kw.test(qb)) return false;

  const tpa = numericQuestionTemplate(qa);
  const tpb = numericQuestionTemplate(qb);
  if (tpa.length < 10 || tpa !== tpb) {
    const ma = extractFirstMonthOrdinal(qa);
    const mb = extractFirstMonthOrdinal(qb);
    if (ma != null && mb != null && ma !== mb && /before|after|by|end of/i.test(qa) && /before|after|by|end of/i.test(qb)) {
      if (stripMonthNames(qa) === stripMonthNames(qb)) return true;
    }
    return false;
  }

  const va = extractPrimaryThresholdValue(qa);
  const vb = extractPrimaryThresholdValue(qb);
  if (va == null || vb == null) return false;
  const rel = Math.abs(va - vb) / Math.max(va, vb, 1);
  return rel > 1e-6;
}

export type StructuralLaneClassification = {
  lane: EquivalenceMicroStructuralLane;
  reason: StructuralAssignmentReason;
};

/**
 * Ordem: partição competitiva → escada monótona (subset) → equivalência lexical forte → resto (exclusive).
 */
export function classifyEquivalenceMicroStructuralLaneWithReason(
  ma: NormalizedMarket,
  mb: NormalizedMarket
): StructuralLaneClassification {
  const qa = ma.question || "";
  const qb = mb.question || "";

  if (isCompetitivePartitionBalanceOfPower(qa, qb)) {
    return { lane: "graph_exclusive_micro", reason: "competitive_partition_balance_of_power" };
  }
  if (isCompetitiveRivalSportsEntities(qa, qb)) {
    return { lane: "graph_exclusive_micro", reason: "competitive_partition_rival_sports_entities" };
  }
  if (isCompetitivePartisanChamberPermutation(qa, qb)) {
    return { lane: "graph_exclusive_micro", reason: "competitive_partition_partisan_chamber_or_grid" };
  }
  if (isCompetitivePartisanChamberGrid(qa, qb)) {
    return { lane: "graph_exclusive_micro", reason: "competitive_partition_partisan_chamber_or_grid" };
  }

  if (isMonotonicThresholdOrDateLadder(qa, qb)) {
    return { lane: "graph_subset_micro", reason: "monotonic_threshold_or_date_ladder" };
  }

  const ta = enrichedStructuralTokens(qa);
  const tb = enrichedStructuralTokens(qb);
  const jac = jaccardSimilarity(ta, tb);
  if (
    jac === 1 &&
    ta.size > 0 &&
    setsEqual(ta, tb) &&
    normEquivalenceSurface(qa) === normEquivalenceSurface(qb)
  ) {
    return { lane: "graph_equivalence_micro", reason: "true_equivalence_identical_informative_tokens" };
  }

  return { lane: "graph_exclusive_micro", reason: "residual_not_pure_equivalence_nor_monotonic_subset" };
}

export function classifyEquivalenceMicroStructuralLane(
  ma: NormalizedMarket,
  mb: NormalizedMarket
): EquivalenceMicroStructuralLane {
  return classifyEquivalenceMicroStructuralLaneWithReason(ma, mb).lane;
}

export type StructuralMicroLaneSample = {
  opportunityId: string;
  type: EquivalenceMicroStructuralLane;
  structuralAssignmentReason?: StructuralAssignmentReason;
  questions: [string, string];
};

export type StructuralMicroLaneScanSnapshot = {
  computedAt: string;
  note: string;
  counts: Record<EquivalenceMicroStructuralLane, number>;
  reasonCounts: Record<StructuralAssignmentReason, number>;
  samples: Record<EquivalenceMicroStructuralLane, StructuralMicroLaneSample[]>;
};

const SAMPLE_CAP = 6;

const ALL_REASONS: StructuralAssignmentReason[] = [
  "competitive_partition_balance_of_power",
  "competitive_partition_rival_sports_entities",
  "competitive_partition_partisan_chamber_or_grid",
  "monotonic_threshold_or_date_ladder",
  "true_equivalence_identical_informative_tokens",
  "residual_not_pure_equivalence_nor_monotonic_subset",
];

function emptyReasonCounts(): Record<StructuralAssignmentReason, number> {
  const o = {} as Record<StructuralAssignmentReason, number>;
  for (const r of ALL_REASONS) o[r] = 0;
  return o;
}

export function buildStructuralMicroLaneScanSnapshot(
  opportunities: Array<{
    id: string;
    type: string;
    structuralMicroLaneReason?: StructuralAssignmentReason;
    marketsInvolved?: Array<{ question: string }>;
  }>
): StructuralMicroLaneScanSnapshot {
  const counts: Record<EquivalenceMicroStructuralLane, number> = {
    graph_equivalence_micro: 0,
    graph_subset_micro: 0,
    graph_exclusive_micro: 0,
  };
  const reasonCounts = emptyReasonCounts();
  const samples: Record<EquivalenceMicroStructuralLane, StructuralMicroLaneSample[]> = {
    graph_equivalence_micro: [],
    graph_subset_micro: [],
    graph_exclusive_micro: [],
  };

  for (const o of opportunities) {
    if (
      o.type !== "graph_equivalence_micro" &&
      o.type !== "graph_subset_micro" &&
      o.type !== "graph_exclusive_micro"
    ) {
      continue;
    }
    const lane = o.type as EquivalenceMicroStructuralLane;
    counts[lane] += 1;
    const reason = o.structuralMicroLaneReason;
    if (reason) reasonCounts[reason] += 1;

    const inv = o.marketsInvolved || [];
    const q0 = inv[0]?.question?.slice(0, 120) ?? "";
    const q1 = inv[1]?.question?.slice(0, 120) ?? "";
    if (samples[lane].length < SAMPLE_CAP) {
      samples[lane].push({
        opportunityId: o.id,
        type: lane,
        structuralAssignmentReason: reason,
        questions: [q0, q1],
      });
    }
  }

  return {
    computedAt: new Date().toISOString(),
    note:
      "Ordem: BoP/rivalidade desportiva/grelha partidária → exclusive_micro; escada monótona de limiar/data (mesmo template #) → subset_micro; tokens informativos enriquecidos idênticos e Jaccard=1 → equivalence_micro; resto → exclusive_micro. Não se optimiza volume.",
    counts,
    reasonCounts,
    samples,
  };
}
