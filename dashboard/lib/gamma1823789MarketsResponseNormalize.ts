/**
 * Normaliza respostas read-only da API Gamma (/markets?id=…) para uma linha de mercado.
 * Não persiste payloads completos — apenas estruturas mínimas para extrair `clobTokenIds`.
 */

export interface GammaMarketPayloadDiagnostic {
  rootKind: string;
  arrayLength?: number;
  topLevelKeys?: string[];
  hasId: boolean;
  hasQuestion: boolean;
  hasTokensField: boolean;
  hasClobTokenIdsField: boolean;
  hasOutcomesField: boolean;
  hasOutcomePricesField: boolean;
}

export class GammaPayloadUnusableShapeError extends Error {
  readonly code = "gamma_payload_unusable_shape" as const;
  readonly diagnostic: GammaMarketPayloadDiagnostic;

  constructor(diagnostic: GammaMarketPayloadDiagnostic) {
    super(`gamma_payload_unusable_shape:${JSON.stringify(diagnostic)}`);
    this.name = "GammaPayloadUnusableShapeError";
    this.diagnostic = diagnostic;
  }
}

function fieldPresence(o: Record<string, unknown>): Pick<
  GammaMarketPayloadDiagnostic,
  "hasId" | "hasQuestion" | "hasTokensField" | "hasClobTokenIdsField" | "hasOutcomesField" | "hasOutcomePricesField"
> {
  return {
    hasId: "id" in o,
    hasQuestion: "question" in o,
    hasTokensField: "tokens" in o,
    hasClobTokenIdsField: "clobTokenIds" in o,
    hasOutcomesField: "outcomes" in o,
    hasOutcomePricesField: "outcomePrices" in o,
  };
}

export function diagnoseGammaMarketJsonRoot(json: unknown): GammaMarketPayloadDiagnostic {
  const rootKind =
    json === null
      ? "null"
      : json === undefined
        ? "undefined"
        : Array.isArray(json)
          ? "array"
          : typeof json;

  const base: GammaMarketPayloadDiagnostic = {
    rootKind,
    hasId: false,
    hasQuestion: false,
    hasTokensField: false,
    hasClobTokenIdsField: false,
    hasOutcomesField: false,
    hasOutcomePricesField: false,
  };

  if (Array.isArray(json)) {
    base.arrayLength = json.length;
    return base;
  }

  if (json && typeof json === "object" && !Array.isArray(json)) {
    const o = json as Record<string, unknown>;
    base.topLevelKeys = Object.keys(o).sort();
    Object.assign(base, fieldPresence(o));
  }

  return base;
}

/** Indício mínimo de que o objeto é uma linha de mercado Gamma (não um envelope opaco). */
function looksLikeMarketRow(o: Record<string, unknown>): boolean {
  return (
    typeof o.id === "string" ||
    typeof o.question === "string" ||
    o.clobTokenIds !== undefined ||
    o.outcomes !== undefined ||
    o.tokens !== undefined ||
    o.outcomePrices !== undefined
  );
}

/**
 * Extrai um único objeto de mercado aceitando os formatos listados nos requisitos.
 * Não expõe o payload bruto — apenas percorre referências.
 */
export function normalizeGamma1823789MarketRow(json: unknown): Record<string, unknown> {
  const diagnosticBase = diagnoseGammaMarketJsonRoot(json);

  const row = unwrapGammaMarketCandidate(json, 0);

  if (row !== null && looksLikeMarketRow(row)) {
    return row;
  }

  if (json && typeof json === "object" && !Array.isArray(json)) {
    Object.assign(diagnosticBase, fieldPresence(json as Record<string, unknown>));
  }

  throw new GammaPayloadUnusableShapeError(diagnosticBase);
}

const MAX_DEPTH = 10;

function unwrapGammaMarketCandidate(json: unknown, depth: number): Record<string, unknown> | null {
  if (depth > MAX_DEPTH) return null;

  if (json === null || json === undefined) return null;

  if (typeof json === "string" || typeof json === "number" || typeof json === "boolean") {
    return null;
  }

  if (Array.isArray(json)) {
    if (json.length === 0) return null;
    const head = json[0];
    if (head && typeof head === "object" && !Array.isArray(head)) {
      const inner = unwrapGammaMarketCandidate(head, depth + 1);
      if (inner) return inner;
    }
    return null;
  }

  const o = json as Record<string, unknown>;

  if (o.market !== undefined && o.market !== null) {
    const inner = unwrapGammaMarketCandidate(o.market, depth + 1);
    if (inner) return inner;
  }

  for (const key of ["data", "markets", "results"] as const) {
    if (key in o) {
      const inner = unwrapGammaMarketCandidate(o[key], depth + 1);
      if (inner) return inner;
    }
  }

  if (looksLikeMarketRow(o)) {
    return o;
  }

  return null;
}
