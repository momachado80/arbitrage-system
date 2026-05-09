/**
 * Paper Portfolio Store — estado canónico do portfolio paper.
 * Persistido em globalThis para o mesmo processo Node ver os mesmos trades
 * em qualquer chunk (loop paper, /api/paper/*), mesmo com HMR em next dev.
 */

import type { PaperGraphDiagnosticProvenance, PaperTrade } from "./paperTypes";
import {
  DEFAULT_PAPER_FEE_BUFFER_PER_LEG,
  getClosedTradeNetRealizedPnL,
} from "./paperRealizedPnlSemantics";
import { persistSafetyClose } from "./paperSafetyHistoricalMemory";

const DEFAULT_STARTING_CAPITAL = 10_000;
const MAX_ACTIVE_TRADES = 100;
const MAX_CLOSED_TRADES = 500;

/** Chave estável para diagnóstico de integridade entre rotas. */
export const PAPER_PORTFOLIO_STORE_RUNTIME_ID = "__paperPortfolioStore_v1";

const GLOBAL_KEY = PAPER_PORTFOLIO_STORE_RUNTIME_ID;

export type GraphOpenProvenanceSample = {
  at: string;
  tradeId: string;
  opportunityId: string;
  /** Valor recebido no payload antes do finalize do store (undefined = ausente). */
  payloadGraphDiagnosticProvenanceAtOpen: PaperGraphDiagnosticProvenance | undefined | null;
  /** Valor gravado em `activeTrades` após `finalizeGraphDiagnosticProvenanceOnTrade`. */
  storedGraphDiagnosticProvenanceAtOpen: PaperGraphDiagnosticProvenance;
  /** true se o store preencheu proveniência (payload ausente ou vazio). */
  coercedAtStore: boolean;
};

type PortfolioIntegrity = {
  lastTradeOpenedAt: string | null;
  lastTradeClosedAt: string | null;
  lastTradeCreatedId: string | null;
  lastTradeRemovedId: string | null;
  lastTradeRemovalReason: string | null;
  /** Última chamada a initPaperPortfolio (não limpa trades; só capital de arranque). */
  lastPortfolioInitAt: string | null;
  storeFirstAttachedAt: string;
  lastOpenRejectedAt: string | null;
  lastOpenRejectReason: string | null;
  /** Última abertura graph: diagnóstico de propagação de proveniência (payload vs valor persistido). */
  lastGraphOpenProvenanceSample: GraphOpenProvenanceSample | null;
};

type Store = {
  startingCapital: number;
  activeTrades: PaperTrade[];
  closedTrades: PaperTrade[];
  peakEquity: number;
  integrity: PortfolioIntegrity;
};

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store | undefined>;
  if (!g[GLOBAL_KEY]) {
    const now = new Date().toISOString();
    g[GLOBAL_KEY] = {
      startingCapital: DEFAULT_STARTING_CAPITAL,
      activeTrades: [],
      closedTrades: [],
      peakEquity: DEFAULT_STARTING_CAPITAL,
      integrity: {
        lastTradeOpenedAt: null,
        lastTradeClosedAt: null,
        lastTradeCreatedId: null,
        lastTradeRemovedId: null,
        lastTradeRemovalReason: null,
        lastPortfolioInitAt: null,
        storeFirstAttachedAt: now,
        lastOpenRejectedAt: null,
        lastOpenRejectReason: null,
        lastGraphOpenProvenanceSample: null,
      },
    };
    if (process.env.NODE_ENV === "development") {
      console.log("[PaperPortfolio] canonical store attached to globalThis (HMR-safe)");
    }
  }
  return g[GLOBAL_KEY]!;
}

function safeNumber(v: unknown, def: number): number {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return def;
}

/** Inferência mínima por tipo de oportunidade (sem dependência de graphOpportunityPaperImpact — evita ciclos). */
function inferGraphProvenanceFromOpportunityType(ot: string): PaperGraphDiagnosticProvenance {
  switch (ot) {
    case "graph_cycle":
      return "cycle";
    case "graph_equivalence":
    case "graph_equivalence_micro":
      return "equivalent";
    case "graph_subset_micro":
      return "subset";
    case "graph_exclusive_micro":
      return "exclusive";
    case "graph_subset":
      return "subset";
    case "graph_exclusive":
      return "exclusive";
    default:
      return "unknown";
  }
}

/**
 * Garante `graphDiagnosticProvenanceAtOpen` em trades graph antes de persistir.
 * Motivo: `JSON.stringify` omite `undefined` na API — o campo tem de ser string concreta.
 * Usa `entryProfileKeyAtOpen` / `opportunityType` se o payload vier sem proveniência.
 */
export function finalizeGraphDiagnosticProvenanceOnTrade(trade: PaperTrade): PaperTrade {
  if (trade.sourceType !== "graph") return trade;
  const raw = trade.graphDiagnosticProvenanceAtOpen;
  const hasPayload =
    raw != null && typeof raw === "string" && raw.length > 0;
  if (hasPayload) return trade;
  const ot =
    trade.entryProfileKeyAtOpen != null && trade.entryProfileKeyAtOpen.includes("|")
      ? (trade.entryProfileKeyAtOpen.split("|")[1] ?? "").trim() || trade.opportunityType
      : trade.opportunityType;
  return {
    ...trade,
    graphDiagnosticProvenanceAtOpen: inferGraphProvenanceFromOpportunityType(String(ot)),
  };
}

/** Mesma regra de contagem que computePaperAnalytics (fechados com PnL + activos). */
function analyticsReadableTradeTotal(st: Store): number {
  const closed = st.closedTrades.filter((t) => t.status === "closed" && t.realizedPnL !== undefined);
  return closed.length + st.activeTrades.length;
}

export function initPaperPortfolio(initialCapital?: number): void {
  const st = getStore();
  st.startingCapital = Math.max(100, safeNumber(initialCapital, DEFAULT_STARTING_CAPITAL));
  st.peakEquity = st.startingCapital;
  st.integrity.lastPortfolioInitAt = new Date().toISOString();
}

export function getPaperPortfolio() {
  const st = getStore();
  const { activeTrades, closedTrades, startingCapital } = st;
  let peakEquity = st.peakEquity;
  const reserved = activeTrades.reduce((s, t) => s + (t.filledCapital || 0), 0);
  const realized = closedTrades.reduce(
    (s, t) => s + getClosedTradeNetRealizedPnL(t, DEFAULT_PAPER_FEE_BUFFER_PER_LEG),
    0
  );
  const unrealized = activeTrades.reduce((s, t) => {
    const entry = t.entryPriceEstimate || 0;
    const mark =
      t.lastMarkPx != null && Number.isFinite(t.lastMarkPx)
        ? t.lastMarkPx
        : 1 - (t.grossEdgeAtEntry || 0);
    const pnl = (t.filledCapital || 0) * ((mark - entry) / Math.max(0.001, entry));
    return s + pnl;
  }, 0);
  const currentEquity = startingCapital + realized + unrealized;
  const available = Math.max(0, currentEquity - reserved);
  const maxDrawdown = peakEquity > 0 ? (peakEquity - Math.min(peakEquity, currentEquity)) / peakEquity : 0;
  if (currentEquity > peakEquity) st.peakEquity = currentEquity;

  const exposureByType: Record<string, number> = {};
  const exposureByCluster: Record<string, number> = {};
  const exposureByMarket: Record<string, number> = {};

  for (const t of activeTrades) {
    const cap = t.filledCapital || 0;
    exposureByType[t.opportunityType] = (exposureByType[t.opportunityType] || 0) + cap;
    if (t.clusterId) exposureByCluster[t.clusterId] = (exposureByCluster[t.clusterId] || 0) + cap;
    for (const m of t.marketsInvolved || []) {
      exposureByMarket[m.marketId] = (exposureByMarket[m.marketId] || 0) + cap;
    }
  }

  return {
    startingCapital,
    currentEquity,
    availableCapital: available,
    reservedCapital: reserved,
    realizedPnL: realized,
    unrealizedPnL: unrealized,
    maxDrawdown,
    activeTrades,
    closedTrades,
    exposureByType,
    exposureByCluster,
    exposureByMarket,
  };
}

export function getActivePaperTrades(): PaperTrade[] {
  return [...getStore().activeTrades];
}

export function getClosedPaperTrades(limit = 50): PaperTrade[] {
  const closedTrades = getStore().closedTrades;
  const start = Math.max(0, closedTrades.length - limit);
  return closedTrades.slice(start);
}

export interface PaperPortfolioSummary {
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  realizedPnL: number;
  unrealizedPnL: number;
  maxDrawdown: number;
  activeTrades: number;
  closedTrades: number;
  exposureByType: Record<string, number>;
  exposureByCluster: Record<string, number>;
  exposureByMarket: Record<string, number>;
}

export function getPaperPortfolioSummary(): PaperPortfolioSummary {
  const p = getPaperPortfolio();
  return {
    startingCapital: p.startingCapital,
    currentEquity: p.currentEquity,
    availableCapital: p.availableCapital,
    reservedCapital: p.reservedCapital,
    realizedPnL: p.realizedPnL,
    unrealizedPnL: p.unrealizedPnL,
    maxDrawdown: p.maxDrawdown,
    activeTrades: p.activeTrades.length,
    closedTrades: p.closedTrades.length,
    exposureByType: p.exposureByType,
    exposureByCluster: p.exposureByCluster,
    exposureByMarket: p.exposureByMarket,
  };
}

export function addActiveTrade(trade: PaperTrade): void {
  const st = getStore();
  try {
    if (st.activeTrades.length >= MAX_ACTIVE_TRADES) {
      console.warn("[PaperPortfolio] Max active trades reached");
      st.integrity.lastOpenRejectedAt = new Date().toISOString();
      st.integrity.lastOpenRejectReason = "max_active_trades";
      return;
    }
    if (!trade?.tradeId || !trade?.opportunityId) return;
    const payloadProv = trade.graphDiagnosticProvenanceAtOpen;
    const finalized = finalizeGraphDiagnosticProvenanceOnTrade(trade);
    if (trade.sourceType === "graph") {
      st.integrity.lastGraphOpenProvenanceSample = {
        at: new Date().toISOString(),
        tradeId: trade.tradeId,
        opportunityId: trade.opportunityId,
        payloadGraphDiagnosticProvenanceAtOpen: payloadProv,
        storedGraphDiagnosticProvenanceAtOpen: finalized.graphDiagnosticProvenanceAtOpen ?? "unknown",
        coercedAtStore: !(
          payloadProv != null && typeof payloadProv === "string" && payloadProv.length > 0
        ),
      };
    }
    st.activeTrades.push({
      ...finalized,
      filledCapital: safeNumber(finalized.filledCapital, 0),
      realizedPnL: 0,
      realizedReturn: 0,
      holdingTimeMs: 0,
      maxAdverseExcursion: safeNumber(finalized.maxAdverseExcursion, 0),
      maxFavorableExcursion: safeNumber(finalized.maxFavorableExcursion, 0),
    });
    const now = new Date().toISOString();
    st.integrity.lastTradeOpenedAt = now;
    st.integrity.lastTradeCreatedId = trade.tradeId;
    st.integrity.lastOpenRejectReason = null;
    st.integrity.lastOpenRejectedAt = null;
  } catch {
    // non-fatal
  }
}

export function getActiveTradeById(tradeId: string): PaperTrade | undefined {
  return getStore().activeTrades.find((t) => t.tradeId === tradeId);
}

export function updateActiveTradeMtm(
  tradeId: string,
  updates: {
    lastMarkPx: number;
    lastMarkAt: string;
    maxAdverseExcursion: number;
    maxFavorableExcursion: number;
  }
): void {
  const t = getStore().activeTrades.find((x) => x.tradeId === tradeId);
  if (!t) return;
  t.lastMarkPx = updates.lastMarkPx;
  t.lastMarkAt = updates.lastMarkAt;
  t.maxAdverseExcursion = updates.maxAdverseExcursion;
  t.maxFavorableExcursion = updates.maxFavorableExcursion;
}

export function closeTrade(tradeId: string, updates: Partial<PaperTrade>): void {
  const st = getStore();
  const idx = st.activeTrades.findIndex((t) => t.tradeId === tradeId);
  if (idx < 0) return;
  const t = st.activeTrades[idx];
  st.activeTrades.splice(idx, 1);
  const closed: PaperTrade = finalizeGraphDiagnosticProvenanceOnTrade({
    ...t,
    ...updates,
    status: "closed",
    closedAt: updates.closedAt || new Date().toISOString(),
  });
  st.closedTrades.push(closed);
  const now = new Date().toISOString();
  st.integrity.lastTradeClosedAt = now;
  st.integrity.lastTradeRemovedId = tradeId;
  st.integrity.lastTradeRemovalReason = "closeTrade_active_to_closed";
  if (st.closedTrades.length > MAX_CLOSED_TRADES) {
    st.closedTrades = st.closedTrades.slice(-MAX_CLOSED_TRADES);
  }
  try {
    persistSafetyClose(closed);
  } catch {
    // non-fatal: safety memory must not break portfolio
  }
}

export function getAvailableCapital(): number {
  const p = getPaperPortfolio();
  return p.availableCapital;
}

export type PaperPortfolioStateIntegrity = {
  storeRuntimeIdentity: string;
  activeTradesStoreCount: number;
  closedTradesStoreCount: number;
  /** Igual à soma usada em computePaperAnalytics.totalTrades para este store. */
  analyticsReadableTradesCount: number;
  lastTradeOpenedAt: string | null;
  lastTradeClosedAt: string | null;
  lastTradeCreatedId: string | null;
  lastTradeRemovedId: string | null;
  lastTradeRemovalReason: string | null;
  /** Não há wipe explícito de trades no código; null salvo extensão futura. */
  lastStoreResetAt: string | null;
  lastPortfolioInitAt: string | null;
  storeFirstAttachedAt: string;
  lastOpenRejectedAt: string | null;
  lastOpenRejectReason: string | null;
  lastGraphOpenProvenanceSample: GraphOpenProvenanceSample | null;
};

export function getPaperPortfolioStateIntegrity(): PaperPortfolioStateIntegrity {
  const st = getStore();
  const i = st.integrity;
  return {
    storeRuntimeIdentity: PAPER_PORTFOLIO_STORE_RUNTIME_ID,
    activeTradesStoreCount: st.activeTrades.length,
    closedTradesStoreCount: st.closedTrades.length,
    analyticsReadableTradesCount: analyticsReadableTradeTotal(st),
    lastTradeOpenedAt: i.lastTradeOpenedAt,
    lastTradeClosedAt: i.lastTradeClosedAt,
    lastTradeCreatedId: i.lastTradeCreatedId,
    lastTradeRemovedId: i.lastTradeRemovedId,
    lastTradeRemovalReason: i.lastTradeRemovalReason,
    lastStoreResetAt: null,
    lastPortfolioInitAt: i.lastPortfolioInitAt,
    storeFirstAttachedAt: i.storeFirstAttachedAt,
    lastOpenRejectedAt: i.lastOpenRejectedAt,
    lastOpenRejectReason: i.lastOpenRejectReason,
    lastGraphOpenProvenanceSample: i.lastGraphOpenProvenanceSample,
  };
}
