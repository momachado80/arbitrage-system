/**
 * Pure picker para o agendamento de catalisadores via ESPN public scoreboard.
 *
 * Razão de ser: a hipótese narrow #1 (post-event reversion) observa janelas
 * POST_EVENT_15M (+120m) e POST_EVENT_60M (+240m), portanto o plano precisa
 * manter o jogo no plano por algumas horas DEPOIS do início do evento — não
 * apenas até EVENT_START. Sem essa janela de retenção, todo rebuild do plano
 * que acontece após `eventStartUtc` reclassifica o mercado como NO_NEAR_CATALYST
 * e apaga as janelas pós-evento, impedindo a coleta da própria hipótese.
 *
 * Funções puras, sem rede e sem I/O. Sem .paper, sem execução, sem microcapital.
 */

/**
 * Janela de retenção pós-início do evento. Cobre POST_EVENT_60M (+240m / 4h)
 * com folga para tolerância do scout. Manter ≥ 4h + tolerância do scout.
 */
export const POST_EVENT_RETENTION_MS = 5 * 60 * 60 * 1000;

export interface NextGamePick {
  eventName: string;
  eventStartUtc: string;
  opponentShort: string | null;
}

/**
 * Escolhe o jogo mais cedo entre os candidatos que se encaixam na janela
 * [now − retentionMs, horizonEnd]. Aceitar jogos iniciados recentemente é o
 * que mantém POST_EVENT_15M/POST_EVENT_60M no plano enquanto ainda no futuro.
 *
 * Retorna null quando nenhum candidato está na janela elegível.
 */
export function pickEarliestEspnGameWithinRetention(
  candidates: ReadonlyArray<NextGamePick>,
  now: Date,
  horizonEnd: Date,
  retentionMs: number = POST_EVENT_RETENTION_MS,
): NextGamePick | null {
  const minT = now.getTime() - retentionMs;
  const maxT = horizonEnd.getTime();
  let best: NextGamePick | null = null;
  let bestT = Infinity;
  for (const pick of candidates) {
    const ts = new Date(pick.eventStartUtc).getTime();
    if (!Number.isFinite(ts)) continue;
    if (ts < minT || ts > maxT) continue;
    if (ts < bestT) {
      bestT = ts;
      best = pick;
    }
  }
  return best;
}
