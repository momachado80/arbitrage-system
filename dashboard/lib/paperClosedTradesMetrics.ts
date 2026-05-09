/**
 * Fonte única de predicados para fechados paper (analytics, proveniência, auditoria).
 * Evita divergência entre snapshot API e store.
 */

import type { PaperTrade } from "./paperTypes";
import {
  getClosedPaperTrades,
  getPaperPortfolio,
  PAPER_PORTFOLIO_STORE_RUNTIME_ID,
} from "./paperPortfolioStore";
import { getPaperSimRuntime, ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY } from "./nodeProcessRuntimeState";
import { hasClosedPaperTradeFinitePnlSignal } from "./paperRealizedPnlSemantics";

/** Limite de fechados em GET /api/paper/trades (env `PAPER_API_RECENT_CLOSED_LIMIT`). */
export function getPaperApiRecentClosedLimit(): number {
  const raw = process.env.PAPER_API_RECENT_CLOSED_LIMIT?.trim();
  if (raw) {
    const n = Number(raw);
    if (Number.isFinite(n) && n >= 1 && n <= 100) return Math.floor(n);
  }
  return 25;
}

/** Fechado com algum sinal PnL finito persistido (net, gross explícito, ou legado). */
export function isClosedTradeWithFiniteRealizedPnl(t: PaperTrade): boolean {
  return hasClosedPaperTradeFinitePnlSignal(t);
}

/** Fechados utilizáveis para métricas (cópia lógica; mesma array subjacente que o store). */
export function getClosedTradesWithFiniteRealizedPnl(): PaperTrade[] {
  return getPaperPortfolio().closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);
}

/**
 * Trade graph para métricas de proveniência: `sourceType` ou perfil/ tipo alinhados ao motor.
 */
export function isGraphTradeForProvenanceMetrics(t: PaperTrade): boolean {
  if (t.sourceType === "graph") return true;
  if (t.entryProfileKeyAtOpen != null && t.entryProfileKeyAtOpen.startsWith("graph|")) return true;
  const ot = t.opportunityType;
  return typeof ot === "string" && ot.startsWith("graph_");
}

export function filterGraphClosedTradesForProvenanceMetrics(trades: PaperTrade[]): PaperTrade[] {
  return trades.filter((t) => isGraphTradeForProvenanceMetrics(t));
}

/**
 * Universo da auditoria `graphProvenanceQualityAudit`: alinhado a
 * `buildGraphProvenancePropagationDiagnostics` (graph + fechado), sem exigir PnL finito.
 * Trades com PnL inválido entram na auditoria com PnL efectivo 0 nos agregados.
 */
export function getClosedGraphTradesForProvenanceQualityAudit(): PaperTrade[] {
  return getPaperPortfolio().closedTrades.filter(
    (t) => t.status === "closed" && t.sourceType === "graph"
  );
}

export type ClosedTradesSourceDiagnostics = {
  /** Total de entradas em `store.closedTrades` (inclui inválidos). */
  storeClosedTradesRawLength: number;
  /** Predicado analytics: status closed + realizedPnL finito. */
  closedTradesWithFinitePnlCount: number;
  /** Subconjunto graph (predicado alargado) sobre fechados com PnL finito. */
  closedGraphTradesCount: number;
  /** Linhas em `recentClosed` de GET /api/paper/trades (= slice últimos N no store). */
  closedTradesCountSeenByTradesRoute: number;
  /** Mesmo universo que `graphProvenanceQualityAudit.closedGraphTradesAnalyzed`. */
  closedTradesCountSeenByQualityAudit: number;
  /** Mesmo universo que `aggregateClosedTradesByGraphProvenance` sobre fechados finitos. */
  closedTradesCountSeenByAnalytics: number;
  /** `true` quando auditoria e analytics contam o mesmo conjunto graph+fechado+finito. */
  qualityAuditAndAnalyticsGraphClosedCountsMatch: boolean;
  /** Último snapshot em memória (só referência; pode estar stale vs store). */
  snapshotRecentClosedLength: number;
  snapshotClosedFinitePnlCount: number;
  snapshotGraphClosedCount: number;
  /** Após alinhar GET /paper/trades ao store, este contador = últimos N fechados finitos no store. */
  recentClosedFiniteCountFromStore: number;
  storeRuntimeIdentity: string;
  processRuntimeRootKey: string;
  /** Identidade útil para confirmar mesmo processo / mesma raiz globalThis. */
  runtimeIdentity: string;
};

export function buildClosedTradesSourceDiagnostics(
  recentClosedLimit: number = getPaperApiRecentClosedLimit()
): ClosedTradesSourceDiagnostics {
  const raw = getPaperPortfolio().closedTrades;
  const finite = getClosedTradesWithFiniteRealizedPnl();
  const graph = filterGraphClosedTradesForProvenanceMetrics(finite);
  const graphClosedForQualityAudit = getClosedGraphTradesForProvenanceQualityAudit();
  const snap = getPaperSimRuntime().tradesSnapshot.recentClosed;
  const snapFinite = snap.filter(isClosedTradeWithFiniteRealizedPnl);
  const snapGraph = filterGraphClosedTradesForProvenanceMetrics(snapFinite);
  const start = Math.max(0, raw.length - recentClosedLimit);
  const recentSlice = raw.slice(start).filter(isClosedTradeWithFiniteRealizedPnl);
  const routeSlice = getClosedPaperTrades(recentClosedLimit);
  let analyticsGraphClosed = 0;
  for (const t of finite) {
    if (isGraphTradeForProvenanceMetrics(t)) analyticsGraphClosed += 1;
  }

  return {
    storeClosedTradesRawLength: raw.length,
    closedTradesWithFinitePnlCount: finite.length,
    closedGraphTradesCount: graph.length,
    closedTradesCountSeenByTradesRoute: routeSlice.length,
    closedTradesCountSeenByQualityAudit: graphClosedForQualityAudit.length,
    closedTradesCountSeenByAnalytics: analyticsGraphClosed,
    qualityAuditAndAnalyticsGraphClosedCountsMatch: graph.length === analyticsGraphClosed,
    snapshotRecentClosedLength: snap.length,
    snapshotClosedFinitePnlCount: snapFinite.length,
    snapshotGraphClosedCount: snapGraph.length,
    recentClosedFiniteCountFromStore: recentSlice.length,
    storeRuntimeIdentity: PAPER_PORTFOLIO_STORE_RUNTIME_ID,
    processRuntimeRootKey: ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY,
    runtimeIdentity: `${PAPER_PORTFOLIO_STORE_RUNTIME_ID}@${ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY}`,
  };
}
