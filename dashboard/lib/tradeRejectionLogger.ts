/**
 * Trade Rejection Logger — structured diagnostics for rejected trades.
 * In-memory rolling buffer; no changes to execution logic.
 */

export type TradeRejectionReason =
  | "INSUFFICIENT_LIQUIDITY"
  | "SLIPPAGE_TOO_HIGH"
  | "EDGE_BELOW_THRESHOLD"
  | "LATENCY_RISK"
  | "TRADE_SIZE_TOO_SMALL"
  | "EXECUTION_ENGINE_DISABLED"
  | "UNKNOWN";

export interface TradeRejectionEvent {
  timestamp: number;
  marketId: string;
  edgeBps: number;
  reason: TradeRejectionReason;
}

const MAX_EVENTS = 5000;
const buffer: TradeRejectionEvent[] = [];

export function logTradeRejection(event: TradeRejectionEvent): void {
  try {
    buffer.push(event);
    if (buffer.length > MAX_EVENTS) {
      buffer.splice(0, buffer.length - MAX_EVENTS);
    }
  } catch {
    // non-fatal
  }
}

export function getRejectionStats(): {
  countsByReason: Record<TradeRejectionReason, number>;
  totalRejected: number;
  recentCount: number;
} {
  const countsByReason: Record<TradeRejectionReason, number> = {
    INSUFFICIENT_LIQUIDITY: 0,
    SLIPPAGE_TOO_HIGH: 0,
    EDGE_BELOW_THRESHOLD: 0,
    LATENCY_RISK: 0,
    TRADE_SIZE_TOO_SMALL: 0,
    EXECUTION_ENGINE_DISABLED: 0,
    UNKNOWN: 0,
  };

  for (const e of buffer) {
    countsByReason[e.reason] = (countsByReason[e.reason] ?? 0) + 1;
  }

  return {
    countsByReason,
    totalRejected: buffer.length,
    recentCount: buffer.length,
  };
}
