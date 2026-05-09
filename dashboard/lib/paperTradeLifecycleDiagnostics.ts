/**
 * Observação do ciclo de vida paper (sem alterar PnL nem preços).
 * globalThis — mesmo padrão que outros diagnósticos.
 */

const GLOBAL_KEY = "__paperTradeLifecycleDiagnostics_v1";
const MAX_TICKS = 400;
const MAX_CLOSES = 80;

export type PaperTradeLifecycleTick = {
  atIso: string;
  tradeId: string;
  opportunityId: string;
  opportunityType: string;
  /** Se a oportunidade existe no lote deste ciclo (oppMap) */
  inOppMap: boolean;
  latestEdge: number | null;
  entryPriceEstimate: number;
  /** 1 - edge se houver latestState efectivo; senão null */
  markPxFromLatest: number | null;
  /** Origem do mark para auditoria: oppMap, MTM por subjacentes, ou nenhum */
  markSource?: "opp_map" | "mtm" | "none";
  /** Mark efectivo alinhado a latestState (1 - edge) quando existe estado */
  effectiveMarkPx?: number | null;
};

export type PaperTradeLifecycleClose = PaperTradeLifecycleTick & {
  exitCondition: string;
  exitPriceEstimate: number;
  realizedPnL: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
  /** true quando latestState era null neste ciclo → simulateExit fixa exit = entry */
  exitEqualsEntryBecauseNoLatest: boolean;
};

type Store = {
  ticks: PaperTradeLifecycleTick[];
  closes: PaperTradeLifecycleClose[];
};

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = { ticks: [], closes: [] };
  return g[GLOBAL_KEY];
}

export function recordPaperTradeLifecycleTick(row: Omit<PaperTradeLifecycleTick, "atIso">): void {
  const st = getStore();
  const tick: PaperTradeLifecycleTick = {
    ...row,
    atIso: new Date().toISOString(),
  };
  st.ticks.push(tick);
  if (st.ticks.length > MAX_TICKS) st.ticks.splice(0, st.ticks.length - MAX_TICKS);
}

export function recordPaperTradeLifecycleClose(
  row: Omit<PaperTradeLifecycleClose, "atIso" | "exitEqualsEntryBecauseNoLatest"> & {
    exitEqualsEntryBecauseNoLatest?: boolean;
  }
): void {
  const st = getStore();
  const exitEqualsEntryBecauseNoLatest =
    row.exitEqualsEntryBecauseNoLatest ??
    (!row.inOppMap || row.markPxFromLatest == null);
  const c: PaperTradeLifecycleClose = {
    ...row,
    atIso: new Date().toISOString(),
    exitEqualsEntryBecauseNoLatest,
  };
  st.closes.push(c);
  if (st.closes.length > MAX_CLOSES) st.closes.splice(0, st.closes.length - MAX_CLOSES);
}

export function getPaperTradeLifecycleDiagnostics(): {
  recentTicks: PaperTradeLifecycleTick[];
  recentCloses: PaperTradeLifecycleClose[];
  summary: {
    tickCount: number;
    closeCount: number;
    closesWithNoLatestAtClose: number;
  };
} {
  const st = getStore();
  const closesWithNoLatestAtClose = st.closes.filter((c) => c.exitEqualsEntryBecauseNoLatest).length;
  return {
    recentTicks: [...st.ticks].slice(-80),
    recentCloses: [...st.closes].slice(-20),
    summary: {
      tickCount: st.ticks.length,
      closeCount: st.closes.length,
      closesWithNoLatestAtClose,
    },
  };
}

/** Buffer completo de fechos instrumentados (cap interno MAX_CLOSES); só leitura para auditorias. */
export function getPaperTradeLifecycleClosesBuffer(): PaperTradeLifecycleClose[] {
  return [...getStore().closes];
}
