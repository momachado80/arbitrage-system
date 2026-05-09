/**
 * Último estado da whitelist PAPER_MARKET_IDS após resolver mercados (scan standard).
 * globalThis — alinhado com outros diagnósticos paper.
 */

import type { NormalizedMarket } from "./polymarketClient";
import { getGammaFetchByIdDiagnostics } from "./gammaFetchByIdDiagnostics";

const GLOBAL_KEY = "__paperWhitelistHealth_v1";

export type WhitelistIdStatus =
  | "open_active"
  | "closed_or_inactive"
  | "other_resolution_failure"
  | "missing_unresolved";

export type PaperWhitelistHealthSnapshot = {
  atIso: string;
  whitelistIdsRead: number;
  openActiveCount: number;
  closedOrInactiveCount: number;
  otherResolutionFailureCount: number;
  missingUnresolvedCount: number;
  /** ids com mercado fechado/inactivo (Gamma ou NormalizedMarket) */
  idsFailedMarketInactiveOrClosed: string[];
  /** ids com outro erro de fetch/normalize (ver gammaFetchByIdDiagnostics) */
  idsOtherFailures: string[];
  /** ids sem mercado resolvido em byId e sem diagnóstico útil */
  idsMissing: string[];
  perId: Array<{ id: string; status: WhitelistIdStatus; detail: string | null }>;
};

type Store = { last: PaperWhitelistHealthSnapshot | null };

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = { last: null };
  return g[GLOBAL_KEY];
}

/**
 * Chamado após o loop de fetch da whitelist, com o mesmo `byId` usado para montar `markets`.
 */
export function recordPaperWhitelistHealthAfterScan(whitelist: Set<string>, byId: Map<string, NormalizedMarket>): void {
  const diag = getGammaFetchByIdDiagnostics().byMarketId;
  const ids = Array.from(whitelist);
  const perId: PaperWhitelistHealthSnapshot["perId"] = [];
  const idsFailedMarketInactiveOrClosed: string[] = [];
  const idsOtherFailures: string[] = [];
  const idsMissing: string[] = [];
  let openActiveCount = 0;
  let closedOrInactiveCount = 0;
  let otherResolutionFailureCount = 0;
  let missingUnresolvedCount = 0;

  for (const id of ids) {
    const m = byId.get(id);
    if (m) {
      if (!m.active || m.closed) {
        closedOrInactiveCount += 1;
        idsFailedMarketInactiveOrClosed.push(id);
        perId.push({
          id,
          status: "closed_or_inactive",
          detail: `active=${m.active} closed=${m.closed} (resolved in subset map)`,
        });
      } else {
        openActiveCount += 1;
        perId.push({ id, status: "open_active", detail: null });
      }
      continue;
    }
    const last = diag[id]?.last;
    if (last?.failureCode === "MARKET_INACTIVE_OR_CLOSED") {
      closedOrInactiveCount += 1;
      idsFailedMarketInactiveOrClosed.push(id);
      perId.push({
        id,
        status: "closed_or_inactive",
        detail: last.detail ?? last.failureCode,
      });
      continue;
    }
    if (last?.outcome === "failure" && last.failureCode) {
      otherResolutionFailureCount += 1;
      idsOtherFailures.push(id);
      perId.push({
        id,
        status: "other_resolution_failure",
        detail: `${last.failureCode}: ${last.detail ?? ""}`,
      });
      continue;
    }
    missingUnresolvedCount += 1;
    idsMissing.push(id);
    perId.push({ id, status: "missing_unresolved", detail: null });
  }

  getStore().last = {
    atIso: new Date().toISOString(),
    whitelistIdsRead: ids.length,
    openActiveCount,
    closedOrInactiveCount,
    otherResolutionFailureCount,
    missingUnresolvedCount,
    idsFailedMarketInactiveOrClosed,
    idsOtherFailures,
    idsMissing,
    perId,
  };
}

export function getPaperWhitelistHealth(): PaperWhitelistHealthSnapshot | null {
  return getStore().last;
}

export function clearPaperWhitelistHealth(): void {
  getStore().last = null;
}
