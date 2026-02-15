/**
 * Market status enum
 * - open: Market is active and accepting trades
 * - resolved: Market has been settled with a final outcome
 * - cancelled: Market was cancelled (no settlement)
 */
export enum MarketStatus {
  OPEN = 'open',
  RESOLVED = 'resolved',
  CANCELLED = 'cancelled',
}

/**
 * Signal status enum
 * - active: Signal is currently valid for trading
 * - expired: Signal has expired (time-based or market closed)
 * - cancelled: Signal was manually cancelled
 * - resolved: Signal has been resolved (market settled)
 */
export enum SignalStatus {
  ACTIVE = 'active',
  EXPIRED = 'expired',
  CANCELLED = 'cancelled',
  RESOLVED = 'resolved',
}

/**
 * Trade direction enum
 * - YES/NO: For binary prediction markets
 * - LONG/SHORT: For continuous markets
 */
export enum TradeDirection {
  YES = 'YES',
  NO = 'NO',
  LONG = 'LONG',
  SHORT = 'SHORT',
}

/**
 * Market outcome enum
 * - won: The position was correct
 * - lost: The position was incorrect
 * - push: Market was cancelled/voided (stake returned)
 */
export enum MarketOutcomeResult {
  WON = 'won',
  LOST = 'lost',
  PUSH = 'push',
}

/**
 * Market categories
 */
export enum MarketCategory {
  POLITICS = 'politics',
  CRYPTO = 'crypto',
  SPORTS = 'sports',
  TECH = 'tech',
  ENTERTAINMENT = 'entertainment',
  FINANCE = 'finance',
  OTHER = 'other',
}
