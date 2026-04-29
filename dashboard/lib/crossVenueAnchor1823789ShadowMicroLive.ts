/**
 * Shadow micro-live: calcula intent de ordem alvo 1x + auditável; nunca submete à API/CLOB.
 */

import type { CrossVenueAnchor1823789ExecutionGateDigest } from "./crossVenueAnchor1823789ExecutionGate";
import { REF_UNIT_SHARES } from "./crossVenueAnchor1823789ExecutionGate";

export const SHADOW_MICRO_LIVE_PROBE_VERSION = "cross-venue-anchor-1823789-micro-live-shadow-v1" as const;

export type ShadowTradeSide = "buy_yes" | "sell_yes" | "flat";

export interface CrossVenueAnchor1823789ShadowTargetLiveOrder {
  marketId: string;
  yesOutcomeTokenId: string;
  tier: "1x";
  refUnitShares: number;
  side: ShadowTradeSide;
  limitPrice: number;
  sizeShares: number;
  estimatedNotionalUsd: number;
  cappedBy: "none" | "notional_cap";
  impliedEdgeVersusFair: number;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

export function deriveShadowTradeSide(gate: CrossVenueAnchor1823789ExecutionGateDigest): ShadowTradeSide {
  const edge = r6(gate.refinedFairValue - gate.polymarketPriceObserved);
  const eps = 1e-9;
  if (edge > eps) return "buy_yes";
  if (edge < -eps) return "sell_yes";
  return "flat";
}

export function buildCrossVenueAnchor1823789ShadowTargetLiveOrder(args: {
  gate: CrossVenueAnchor1823789ExecutionGateDigest;
  gatesPass: boolean;
  maxNotionalUsd: number;
}): CrossVenueAnchor1823789ShadowTargetLiveOrder | null {
  const { gate, gatesPass, maxNotionalUsd } = args;
  if (!gatesPass) return null;

  const side = deriveShadowTradeSide(gate);
  if (side === "flat") return null;

  const limitPrice =
    side === "buy_yes" ? Math.max(gate.clobBestAsk, 1e-9) : Math.max(gate.clobBestBid, 1e-9);

  let sizeShares = REF_UNIT_SHARES;
  let cappedBy: "none" | "notional_cap" = "none";
  const maxByCap = Math.floor(maxNotionalUsd / limitPrice);
  if (Number.isFinite(maxByCap) && maxByCap >= 0 && maxByCap < sizeShares) {
    sizeShares = maxByCap;
    cappedBy = "notional_cap";
  }
  if (sizeShares <= 0) return null;

  const estimatedNotionalUsd = r6(sizeShares * limitPrice);

  return {
    marketId: gate.marketId,
    yesOutcomeTokenId: gate.yesOutcomeTokenId,
    tier: "1x",
    refUnitShares: REF_UNIT_SHARES,
    side,
    limitPrice: r6(limitPrice),
    sizeShares,
    estimatedNotionalUsd,
    cappedBy,
    impliedEdgeVersusFair: r6(gate.refinedFairValue - gate.polymarketPriceObserved),
  };
}
