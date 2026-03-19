/**
 * Profile Eligibility Funnel — diagnóstico de funil de elegibilidade por profile.
 * Instrumentação puramente observacional para descobrir onde a família estrutural
 * morre economicamente no fluxo. Não altera lógica econômica.
 */

export interface ProfileFunnelState {
  cyclesProcessed: number;
  rawOpportunitiesSeen: number;
  pairEligibleCount: number;
  fillEligibleCount: number;
  capfloorEligibleCount: number;
  degratioEligibleCount: number;
  finalCandidateCount: number;
  openAttemptCount: number;
  openedTradeCount: number;
  closedTradeCount: number;
  realizedPnlTotal: number;
  realizedPnlAvg: number;
}

const funnelState = new Map<string, ProfileFunnelState>();

function ensureProfile(profileId: string): ProfileFunnelState {
  let s = funnelState.get(profileId);
  if (!s) {
    s = {
      cyclesProcessed: 0,
      rawOpportunitiesSeen: 0,
      pairEligibleCount: 0,
      fillEligibleCount: 0,
      capfloorEligibleCount: 0,
      degratioEligibleCount: 0,
      finalCandidateCount: 0,
      openAttemptCount: 0,
      openedTradeCount: 0,
      closedTradeCount: 0,
      realizedPnlTotal: 0,
      realizedPnlAvg: 0,
    };
    funnelState.set(profileId, s);
  }
  return s;
}

/** Chamado uma vez por profile por ciclo, no início do processamento do profile */
export function recordCycleProcessed(profileId: string): void {
  const s = ensureProfile(profileId);
  s.cyclesProcessed++;
}

/** Chamado no início da avaliação de cada (profile, opportunity) — antes de qualquer continue */
export function recordRawOpportunitySeen(profileId: string): void {
  const s = ensureProfile(profileId);
  s.rawOpportunitiesSeen++;
}

/** Chamado quando passa no filtro de pair set (structural risk) */
export function recordPairEligible(profileId: string): void {
  const s = ensureProfile(profileId);
  s.pairEligibleCount++;
}

/** Chamado quando passa no filtro de fill bucket */
export function recordFillEligible(profileId: string): void {
  const s = ensureProfile(profileId);
  s.fillEligibleCount++;
}

/** Chamado quando passa no filtro de capfloor */
export function recordCapfloorEligible(profileId: string): void {
  const s = ensureProfile(profileId);
  s.capfloorEligibleCount++;
}

/** Chamado quando passa no filtro de degratio */
export function recordDegratioEligible(profileId: string): void {
  const s = ensureProfile(profileId);
  s.degratioEligibleCount++;
}

/** Chamado imediatamente antes de addShadowTrade — passou todos os gates */
export function recordFinalCandidate(profileId: string): void {
  const s = ensureProfile(profileId);
  s.finalCandidateCount++;
}

/** Chamado imediatamente antes de addShadowTrade */
export function recordOpenAttempt(profileId: string): void {
  const s = ensureProfile(profileId);
  s.openAttemptCount++;
}

/** Chamado após addShadowTrade bem-sucedido */
export function recordOpened(profileId: string): void {
  const s = ensureProfile(profileId);
  s.openedTradeCount++;
}

/** Atualiza closedTradeCount e realizedPnL a partir do estado atual do store */
export function syncFromStore(
  profileId: string,
  closedCount: number,
  realizedPnlTotal: number
): void {
  const s = ensureProfile(profileId);
  s.closedTradeCount = closedCount;
  s.realizedPnlTotal = realizedPnlTotal;
  s.realizedPnlAvg = closedCount > 0 ? realizedPnlTotal / closedCount : 0;
}

export interface ProfileEligibilityDiagnostics {
  profileId: string;
  label?: string;
  cyclesProcessed: number;
  rawOpportunitiesSeen: number;
  funnel: {
    pairEligibleCount: number;
    fillEligibleCount: number;
    capfloorEligibleCount: number;
    degratioEligibleCount: number;
    finalCandidateCount: number;
    openAttemptCount: number;
    openedTradeCount: number;
    closedTradeCount: number;
  };
  discardReasonCounts: Record<string, number>;
  conversions: {
    rawToPair: number;
    pairToFill: number;
    fillToCapfloor: number;
    capfloorToDegratio: number;
    degratioToCandidate: number;
    candidateToOpened: number;
    openedToClosed: number;
  };
  rates: {
    openRate: number;
    closeRate: number;
  };
  economics: {
    realizedPnlTotal: number;
    realizedPnlAvg: number;
  };
  chokePointSummary: string;
}

/** Também considera rejections do store para choke point em perfis sem filtros estruturais */
function computeChokePointSummary(
  state: ProfileFunnelState,
  discardReasonCounts: Record<string, number>
): string {
  const hasEconomicHistory = state.closedTradeCount > 0 || state.realizedPnlTotal !== 0;
  if (state.cyclesProcessed === 0) {
    if (hasEconomicHistory) {
      return "sem counters de funil materializados, apesar de histórico econômico existente";
    }
    return "profile novo ainda sem dados";
  }
  if (state.rawOpportunitiesSeen === 0) {
    return "profile processado com choke: nenhuma oportunidade bruta vista";
  }
  if (state.pairEligibleCount === 0 && discardReasonCounts["structural_risk_pair_mismatch"] != null) {
    return "profile processado com choke em pair set";
  }
  if (state.fillEligibleCount === 0 && state.pairEligibleCount > 0) {
    return "profile processado com choke em fill bucket";
  }
  if (state.capfloorEligibleCount === 0 && state.fillEligibleCount > 0) {
    return "profile processado com choke em capfloor";
  }
  if (state.degratioEligibleCount === 0 && state.capfloorEligibleCount > 0) {
    return "profile processado com choke em degratio";
  }
  if (state.finalCandidateCount === 0 && state.degratioEligibleCount > 0) {
    return "profile processado com choke em gates pós-structural";
  }
  const topDiscard = Object.entries(discardReasonCounts)
    .sort((a, b) => b[1] - a[1])[0];
  if (topDiscard && topDiscard[1] > state.finalCandidateCount) {
    return `profile processado com choke: principal descarte ${topDiscard[0]} (${topDiscard[1]})`;
  }
  if (state.openedTradeCount === 0 && state.finalCandidateCount > 0) {
    return "profile processado com choke: chegou a candidate mas não abriu (bug?)";
  }
  if (state.closedTradeCount === 0 && state.openedTradeCount > 0) {
    return "profile processado: abriu mas nenhum close ainda";
  }
  if (state.realizedPnlTotal < 0 && state.closedTradeCount > 0) {
    return "profile processado: economia destruída no close (PnL negativo)";
  }
  return "profile processado: fluxo ativo";
}

/** Mescla discardReasonCounts do funnel com rejectionCounts do store para diagnóstico completo */
export function getProfileEligibilityDiagnostics(
  profileIds: string[],
  getLabel: (profileId: string) => string | undefined,
  getClosedCount: (profileId: string) => number,
  getRealizedPnl: (profileId: string) => number,
  rejectionCountsByProfile: Record<string, Record<string, number>>
): Record<string, ProfileEligibilityDiagnostics> {
  const out: Record<string, ProfileEligibilityDiagnostics> = {};
  const allIds = new Set(profileIds);
  for (const pid of Array.from(funnelState.keys())) allIds.add(pid);

  for (const pid of Array.from(allIds)) {
    ensureProfile(pid);
    const closedCount = getClosedCount(pid);
    const realizedPnl = getRealizedPnl(pid);
    syncFromStore(pid, closedCount, realizedPnl);
    const s = funnelState.get(pid)!;
    const discardReasonCounts = rejectionCountsByProfile[pid] ?? {};

    const p2f = s.pairEligibleCount > 0 ? s.fillEligibleCount / s.pairEligibleCount : 0;
    const f2c = s.fillEligibleCount > 0 ? s.capfloorEligibleCount / s.fillEligibleCount : 0;
    const c2d = s.capfloorEligibleCount > 0 ? s.degratioEligibleCount / s.capfloorEligibleCount : 0;
    const d2can = s.degratioEligibleCount > 0 ? s.finalCandidateCount / s.degratioEligibleCount : 0;
    const can2open = s.finalCandidateCount > 0 ? s.openedTradeCount / s.finalCandidateCount : 0;
    const open2close = s.openedTradeCount > 0 ? s.closedTradeCount / s.openedTradeCount : 0;
    const raw2pair = s.rawOpportunitiesSeen > 0 ? s.pairEligibleCount / s.rawOpportunitiesSeen : 0;

    out[pid] = {
      profileId: pid,
      label: getLabel(pid),
      cyclesProcessed: s.cyclesProcessed,
      rawOpportunitiesSeen: s.rawOpportunitiesSeen,
      funnel: {
        pairEligibleCount: s.pairEligibleCount,
        fillEligibleCount: s.fillEligibleCount,
        capfloorEligibleCount: s.capfloorEligibleCount,
        degratioEligibleCount: s.degratioEligibleCount,
        finalCandidateCount: s.finalCandidateCount,
        openAttemptCount: s.openAttemptCount,
        openedTradeCount: s.openedTradeCount,
        closedTradeCount: s.closedTradeCount,
      },
      discardReasonCounts,
      conversions: {
        rawToPair: raw2pair,
        pairToFill: p2f,
        fillToCapfloor: f2c,
        capfloorToDegratio: c2d,
        degratioToCandidate: d2can,
        candidateToOpened: can2open,
        openedToClosed: open2close,
      },
      rates: {
        openRate: s.rawOpportunitiesSeen > 0 ? s.openedTradeCount / s.rawOpportunitiesSeen : 0,
        closeRate: s.openedTradeCount > 0 ? s.closedTradeCount / s.openedTradeCount : 0,
      },
      economics: {
        realizedPnlTotal: s.realizedPnlTotal,
        realizedPnlAvg: s.realizedPnlAvg,
      },
      chokePointSummary: computeChokePointSummary(s, discardReasonCounts),
    };
  }
  return out;
}

/** Retorna resumo compacto do funil para endpoint de profiles */
export function getProfileFunnelSummary(
  profileId: string,
  closedCount: number,
  realizedPnl: number,
  discardReasonCounts?: Record<string, number>
): {
  cyclesProcessed: number;
  rawOpportunitiesSeen: number;
  finalCandidateCount: number;
  openedTradeCount: number;
  closedTradeCount: number;
  realizedPnlTotal: number;
  realizedPnlAvg: number;
  chokePointSummary: string;
} {
  const s = funnelState.get(profileId);
  if (!s) {
    return {
      cyclesProcessed: 0,
      rawOpportunitiesSeen: 0,
      finalCandidateCount: 0,
      openedTradeCount: 0,
      closedTradeCount: closedCount,
      realizedPnlTotal: realizedPnl,
      realizedPnlAvg: closedCount > 0 ? realizedPnl / closedCount : 0,
      chokePointSummary: "sem dados de funil",
    };
  }
  syncFromStore(profileId, closedCount, realizedPnl);
  const updated = funnelState.get(profileId)!;
  return {
    cyclesProcessed: updated.cyclesProcessed,
    rawOpportunitiesSeen: updated.rawOpportunitiesSeen,
    finalCandidateCount: updated.finalCandidateCount,
    openedTradeCount: updated.openedTradeCount,
    closedTradeCount: updated.closedTradeCount,
    realizedPnlTotal: updated.realizedPnlTotal,
    realizedPnlAvg: updated.realizedPnlAvg,
    chokePointSummary: computeChokePointSummary(updated, discardReasonCounts ?? {}),
  };
}
