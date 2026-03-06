/**
 * Market Impact Model — estimates price worsening from requested capital.
 * Conservative, nonlinear impact; low-liquidity opportunities penalized more.
 */

export type LatencyProfile = "fast" | "normal" | "conservative";

export type ImpactConfig = MarketImpactConfig;

export interface MarketImpactConfig {
  impactAlpha: number;
  spreadComponentWeight: number;
  liquidityStressPenalty: number;
  minLiquidityProxy: number;
}

const DEFAULT_CONFIG: MarketImpactConfig = {
  impactAlpha: 1.25,
  spreadComponentWeight: 0.5,
  liquidityStressPenalty: 0.15,
  minLiquidityProxy: 1,
};

export interface OpportunityForImpact {
  liquidity: number;
  spread: number;
  confidence: number;
  edge: number;
  opportunityType?: string;
}

export interface ImpactEstimate {
  requestedCapital: number;
  bookLiquidityProxy: number;
  spread: number;
  impactRate: number;
  expectedPriceWorseningEntry: number;
  expectedPriceWorseningExit: number;
  postImpactNetEdge: number;
}

function liquidityProxy(liq: number, haircut = 0.7): number {
  return Math.max(1, liq * haircut);
}

export function estimateEntryImpact(
  opportunity: OpportunityForImpact,
  requestedCapital: number,
  config?: Partial<MarketImpactConfig>
): ImpactEstimate {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const liq = Math.max(cfg.minLiquidityProxy, opportunity.liquidity);
  const bookProxy = liquidityProxy(liq);
  const spread = Math.max(0.01, opportunity.spread);
  const edge = Math.max(0, opportunity.edge);

  const sizeRatio = requestedCapital / bookProxy;
  const impactFactor = Math.pow(Math.min(sizeRatio, 2), cfg.impactAlpha);
  const impactRate = impactFactor * (spread / bookProxy) * 100;

  const spreadComponent = spread * cfg.spreadComponentWeight * (requestedCapital / bookProxy);
  const impactComponent = spread * impactFactor * 0.3;
  const entryWorsening = Math.min(0.15, spreadComponent + impactComponent);

  const exitWorsening = entryWorsening + cfg.liquidityStressPenalty * (liq < 500 ? 0.02 : 0);

  const totalWorsening = entryWorsening + exitWorsening;
  const postImpactNetEdge = Math.max(0, edge - totalWorsening);

  return {
    requestedCapital,
    bookLiquidityProxy: bookProxy,
    spread,
    impactRate,
    expectedPriceWorseningEntry: entryWorsening,
    expectedPriceWorseningExit: exitWorsening,
    postImpactNetEdge,
  };
}

export interface ExitImpactResult {
  expectedPriceWorseningExit: number;
  exitImpactRate: number;
}

export function estimateExitImpact(
  opportunity: OpportunityForImpact,
  filledCapital: number,
  config?: Partial<MarketImpactConfig>
): ExitImpactResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const bookProxy = liquidityProxy(Math.max(cfg.minLiquidityProxy, opportunity.liquidity));
  const spread = Math.max(0.01, opportunity.spread);

  const sizeRatio = filledCapital / bookProxy;
  const impactFactor = Math.pow(Math.min(sizeRatio, 2), cfg.impactAlpha);
  const exitWorsening = spread * cfg.spreadComponentWeight * sizeRatio +
    spread * impactFactor * 0.2 +
    (opportunity.liquidity < 500 ? cfg.liquidityStressPenalty : 0);

  return {
    expectedPriceWorseningExit: Math.min(0.12, exitWorsening),
    exitImpactRate: impactFactor * (spread / bookProxy) * 100,
  };
}

export function estimatePostImpactNetEdge(
  opportunity: OpportunityForImpact,
  requestedCapital: number,
  config?: Partial<MarketImpactConfig>
): number {
  const est = estimateEntryImpact(opportunity, requestedCapital, config);
  return est.postImpactNetEdge;
}
