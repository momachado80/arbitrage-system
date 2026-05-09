/**
 * Whitelist operacional adaptativa para paper: subset inicial (PAPER_MARKET_IDS)
 * com exclusão automática de ids fechados/inactivos e reposição conservadora do cache global.
 */

import type { NormalizedMarket } from "./polymarketClient";

const GLOBAL_KEY = "__paperAdaptiveWhitelist_v1";

export type PaperAdaptiveWhitelistMode =
  | "nominal"
  | "adapted_excluded_bad_ids"
  | "replenished"
  | "degraded_insufficient_markets";

export type PaperAdaptiveWhitelistDiagnostics = {
  atIso: string;
  /** Tamanho de PAPER_MARKET_IDS (env) neste ciclo. */
  originalWhitelistCount: number;
  /** Ids com mercado activo e aberto usados no scan + filtro. */
  validOperationalCount: number;
  /** Removidos: resolvidos mas closed ou !active. */
  excludedClosedOrInactiveIds: string[];
  /** Removidos: sem NormalizedMarket em byId após o fetch da whitelist. */
  excludedUnresolvedIds: string[];
  /** Adicionados do cache getAllMarkets (liquidez desc.) para atingir o mínimo. */
  replacementIdsAdded: string[];
  mode: PaperAdaptiveWhitelistMode;
  /** Alvo mínimo de ids operacionais (env PAPER_WHITELIST_MIN_OPEN_IDS). */
  minOpenIdsTarget: number;
  /** Resumo legível para logs / UI. */
  explain: string;
};

type Store = { last: PaperAdaptiveWhitelistDiagnostics | null };

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = { last: null };
  return g[GLOBAL_KEY];
}

function envNum(name: string, defaultValue: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return defaultValue;
  const n = Number(raw);
  return Number.isFinite(n) ? n : defaultValue;
}

/** Mínimo de mercados abertos desejados no subset operacional (antes/depois de reposição). */
export function paperWhitelistMinOpenIds(): number {
  const n = envNum("PAPER_WHITELIST_MIN_OPEN_IDS", 3);
  return Math.max(1, Math.min(20, Math.floor(n)));
}

/** Máximo de ids de reposição por ciclo (evita explosão do subset). */
function paperWhitelistMaxReplenishPerCycle(): number {
  const n = envNum("PAPER_WHITELIST_MAX_REPLENISH_PER_CYCLE", 5);
  return Math.max(0, Math.min(20, Math.floor(n)));
}

/**
 * A partir do env whitelist e do mapa já populado (cache + fetch por id),
 * produz o Set usado no scan e no filtro de oportunidades.
 */
export function buildPaperOperationalWhitelist(
  envWhitelist: Set<string>,
  byId: Map<string, NormalizedMarket>,
  allMarkets: NormalizedMarket[]
): { operational: Set<string>; diagnostics: PaperAdaptiveWhitelistDiagnostics } {
  const originalWhitelistCount = envWhitelist.size;
  const excludedClosedOrInactiveIds: string[] = [];
  const excludedUnresolvedIds: string[] = [];
  const operational = new Set<string>();

  for (const id of Array.from(envWhitelist)) {
    const m = byId.get(id);
    if (!m) {
      excludedUnresolvedIds.push(id);
      continue;
    }
    if (!m.active || m.closed) {
      excludedClosedOrInactiveIds.push(id);
      continue;
    }
    operational.add(id);
  }

  const minOpen = paperWhitelistMinOpenIds();
  const maxAdd = paperWhitelistMaxReplenishPerCycle();
  const replacementIdsAdded: string[] = [];

  if (operational.size < minOpen && maxAdd > 0) {
    const need = minOpen - operational.size;
    const cap = Math.min(need, maxAdd);
    const candidates = allMarkets
      .filter((m) => m.active && !m.closed && !operational.has(m.id))
      .sort((a, b) => b.liquidity - a.liquidity);
    for (let i = 0; i < candidates.length && replacementIdsAdded.length < cap; i++) {
      operational.add(candidates[i].id);
      replacementIdsAdded.push(candidates[i].id);
    }
  }

  let mode: PaperAdaptiveWhitelistMode;
  if (operational.size < minOpen) {
    mode = "degraded_insufficient_markets";
  } else if (replacementIdsAdded.length > 0) {
    mode = "replenished";
  } else if (excludedClosedOrInactiveIds.length > 0 || excludedUnresolvedIds.length > 0) {
    mode = "adapted_excluded_bad_ids";
  } else {
    mode = "nominal";
  }

  const explain =
    mode === "nominal"
      ? "Todos os ids do env estão abertos/activos; sem reposição."
      : mode === "adapted_excluded_bad_ids"
        ? `Exclusão automática: ${excludedClosedOrInactiveIds.length} fechados/inactivos, ${excludedUnresolvedIds.length} sem mercado resolvido; subset ainda ≥ mínimo.`
        : mode === "replenished"
          ? `Reposição conservadora: +${replacementIdsAdded.length} mercados do cache (ordenados por liquidez) para atingir ≥${minOpen} ids operacionais.`
          : `Degradado: apenas ${operational.size} mercado(s) operacional(is) após exclusão/reposição (alvo ${minOpen}); cache pode estar vazio ou pouco líquido.`;

  const diagnostics: PaperAdaptiveWhitelistDiagnostics = {
    atIso: new Date().toISOString(),
    originalWhitelistCount,
    validOperationalCount: operational.size,
    excludedClosedOrInactiveIds,
    excludedUnresolvedIds,
    replacementIdsAdded,
    mode,
    minOpenIdsTarget: minOpen,
    explain,
  };

  getStore().last = diagnostics;

  return { operational, diagnostics };
}

export function getPaperAdaptiveWhitelistDiagnostics(): PaperAdaptiveWhitelistDiagnostics | null {
  return getStore().last;
}

export function clearPaperAdaptiveWhitelistDiagnostics(): void {
  getStore().last = null;
}
