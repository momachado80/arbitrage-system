/**
 * Última (e primeira falha) tentativa de GET /markets/{id} por marketId.
 * globalThis — mesmo padrão que paperOpenDiagnostics.
 */

const GLOBAL_KEY = "__gammaFetchByIdDiagnostics_v1";

export type GammaByIdOutcome = "success" | "failure";

export type GammaByIdFailureCode =
  | "TIMEOUT_OR_ABORT"
  | "NETWORK_ERROR"
  | "HTTP_NON_OK"
  | "HTTP_429_RATE_LIMIT"
  | "JSON_PARSE_ERROR"
  | "JSON_NOT_OBJECT"
  | "NORMALIZE_REJECTED"
  | "MARKET_INACTIVE_OR_CLOSED"
  | "UNKNOWN_ERROR";

export type GammaByIdAttempt = {
  atIso: string;
  outcome: GammaByIdOutcome;
  httpRequested: boolean;
  httpStatus: number | null;
  failureCode: GammaByIdFailureCode | null;
  /** Detalhe curto: mensagem de erro, snippet de corpo, ou hint de normalize */
  detail: string | null;
  jsonReceived: boolean;
  jsonWasObject: boolean;
  normalizeReturnedNull: boolean;
  marketActive: boolean | null;
  marketClosed: boolean | null;
};

type Entry = {
  last: GammaByIdAttempt;
  /** Primeira falha registada para este id neste processo */
  firstFailure: GammaByIdAttempt | null;
};

type Store = {
  byMarketId: Record<string, Entry>;
};

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = { byMarketId: {} };
  }
  return g[GLOBAL_KEY];
}

export function recordGammaFetchByIdAttempt(marketId: string, attempt: Omit<GammaByIdAttempt, "atIso"> & { atIso?: string }): void {
  const st = getStore();
  const full: GammaByIdAttempt = {
    ...attempt,
    atIso: attempt.atIso ?? new Date().toISOString(),
  };
  const prev = st.byMarketId[marketId];
  const firstFailure =
    prev?.firstFailure ??
    (full.outcome === "failure" ? full : null);
  st.byMarketId[marketId] = { last: full, firstFailure };
}

export function getGammaFetchByIdDiagnostics(): {
  byMarketId: Record<string, Entry>;
  summaryFromLastAttempts: {
    totalIds: number;
    lastSuccessCount: number;
    lastFailureCount: number;
    byFailureCode: Partial<Record<GammaByIdFailureCode, number>>;
  };
} {
  const st = getStore();
  const byFailureCode: Partial<Record<GammaByIdFailureCode, number>> = {};
  let lastSuccessCount = 0;
  let lastFailureCount = 0;
  for (const e of Object.values(st.byMarketId)) {
    if (e.last.outcome === "success") lastSuccessCount += 1;
    else {
      lastFailureCount += 1;
      const c = e.last.failureCode;
      if (c) byFailureCode[c] = (byFailureCode[c] ?? 0) + 1;
    }
  }
  return {
    byMarketId: { ...st.byMarketId },
    summaryFromLastAttempts: {
      totalIds: Object.keys(st.byMarketId).length,
      lastSuccessCount,
      lastFailureCount,
      byFailureCode,
    },
  };
}
