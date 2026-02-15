import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Execution } from '../executions/entities/execution.entity';
import { MarketOutcome } from '../market-outcomes/entities/market-outcome.entity';
import {
  StatsQueryDto,
  OverviewStatsDto,
  CategoryStatsDto,
  EquityCurveDto,
  EquityCurvePointDto,
  ResolvedTrade,
} from './dto/stats.dto';

@Injectable()
export class StatsService {
  constructor(
    @InjectRepository(Execution)
    private readonly executionRepository: Repository<Execution>,
    @InjectRepository(MarketOutcome)
    private readonly marketOutcomeRepository: Repository<MarketOutcome>,
  ) {}

  // ============================================
  // PUBLIC API METHODS
  // ============================================

  /**
   * Get overview statistics for all trades
   */
  async getOverviewStats(query: StatsQueryDto): Promise<OverviewStatsDto> {
    const trades = await this.getResolvedTrades(query);
    return this.computeOverviewMetrics(trades);
  }

  /**
   * Get statistics grouped by market category
   */
  async getStatsByCategory(query: StatsQueryDto): Promise<CategoryStatsDto[]> {
    const trades = await this.getResolvedTrades(query);
    return this.computeCategoryMetrics(trades);
  }

  /**
   * Get equity curve and max drawdown
   */
  async getEquityCurve(query: StatsQueryDto): Promise<EquityCurveDto> {
    const trades = await this.getResolvedTrades(query);
    return this.computeEquityCurveAndDrawdown(trades);
  }

  // ============================================
  // DATA FETCHING
  // ============================================

  /**
   * Fetch all executions that have a corresponding market outcome (resolved trades).
   * 
   * IMPORTANT: Open trades (without outcome) are EXCLUDED from stats.
   * This ensures all metrics are based on confirmed results only.
   * 
   * The query joins:
   * - Execution → Signal → Market (for category, title)
   * - Market → MarketOutcome (for outcome and finalPrice)
   */
  private async getResolvedTrades(query: StatsQueryDto): Promise<ResolvedTrade[]> {
    const { fromDate, toDate } = query;

    // Build query to get executions with their outcomes
    const queryBuilder = this.executionRepository
      .createQueryBuilder('execution')
      .innerJoinAndSelect('execution.signal', 'signal')
      .innerJoinAndSelect('signal.market', 'market')
      .innerJoin(
        MarketOutcome,
        'outcome',
        'outcome.marketId = market.id',
      )
      .addSelect(['outcome.outcome', 'outcome.finalPrice', 'outcome.resolvedAt']);

    // Apply date filters on execution date
    if (fromDate) {
      queryBuilder.andWhere('execution.executedAt >= :fromDate', {
        fromDate: new Date(fromDate),
      });
    }

    if (toDate) {
      queryBuilder.andWhere('execution.executedAt <= :toDate', {
        toDate: new Date(toDate),
      });
    }

    // Order by execution date for equity curve
    queryBuilder.orderBy('execution.executedAt', 'ASC');

    // Get raw results to include outcome data
    const rawResults = await queryBuilder.getRawAndEntities();

    // Map to ResolvedTrade objects
    const trades: ResolvedTrade[] = [];

    for (let i = 0; i < rawResults.entities.length; i++) {
      const execution = rawResults.entities[i];
      const raw = rawResults.raw[i];

      const outcome = raw.outcome_outcome as 'won' | 'lost' | 'push';
      const finalPrice = parseFloat(raw.outcome_finalPrice) || 0;
      const entryPriceReal = parseFloat(String(execution.entryPriceReal)) || 0;
      const stakeAmount = parseFloat(String(execution.stakeAmount)) || 0;

      // Calculate return for this trade
      const { returnPercent, profitLoss } = this.computeTradeReturn(
        outcome,
        entryPriceReal,
        finalPrice,
        stakeAmount,
      );

      trades.push({
        executionId: execution.id,
        signalId: execution.signalId,
        marketId: execution.signal.marketId,
        marketTitle: execution.signal.market.title,
        category: execution.signal.market.category,
        executedAt: execution.executedAt,
        entryPriceReal,
        stakeAmount,
        outcome,
        finalPrice,
        returnPercent,
        profitLoss,
      });
    }

    return trades;
  }

  // ============================================
  // CALCULATION HELPERS
  // ============================================

  /**
   * Calculate return for a single trade.
   * 
   * RETURN CALCULATION LOGIC:
   * - For prediction markets (binary outcome 0 or 1):
   *   - If won: return = (1 / entryPrice) - 1
   *     Example: bought at 0.40, won → return = (1/0.40) - 1 = 1.5 = 150%
   *   - If lost: return = -1 (lost entire stake)
   *   - If push: return = 0 (stake returned)
   * 
   * - For continuous markets (using finalPrice):
   *   - return = (finalPrice / entryPrice) - 1
   * 
   * We use the simplified binary model for prediction markets.
   */
  private computeTradeReturn(
    outcome: 'won' | 'lost' | 'push',
    entryPriceReal: number,
    finalPrice: number,
    stakeAmount: number,
  ): { returnPercent: number; profitLoss: number } {
    let returnPercent: number;

    switch (outcome) {
      case 'won':
        // Binary market: bought YES at entryPrice, resolved to 1
        // Payout = stake / entryPrice, Profit = Payout - stake
        // Return % = (1 / entryPrice) - 1
        if (entryPriceReal > 0 && entryPriceReal < 1) {
          returnPercent = (1 / entryPriceReal) - 1;
        } else if (finalPrice > 0 && entryPriceReal > 0) {
          // Fallback to finalPrice/entryPrice for non-binary
          returnPercent = (finalPrice / entryPriceReal) - 1;
        } else {
          returnPercent = 0;
        }
        break;

      case 'lost':
        // Lost entire stake
        returnPercent = -1;
        break;

      case 'push':
        // Break-even, stake returned
        returnPercent = 0;
        break;

      default:
        returnPercent = 0;
    }

    const profitLoss = stakeAmount * returnPercent;

    return { returnPercent, profitLoss };
  }

  /**
   * Compute overall metrics from a list of resolved trades
   */
  private computeOverviewMetrics(trades: ResolvedTrade[]): OverviewStatsDto {
    const totalTrades = trades.length;

    if (totalTrades === 0) {
      return {
        totalTrades: 0,
        totalWinningTrades: 0,
        totalLosingTrades: 0,
        totalPushTrades: 0,
        overallWinRate: 0,
        averageRoiPerTrade: 0,
        expectancyPerTrade: 0,
        totalStakeAmount: 0,
        totalProfitLoss: 0,
      };
    }

    const totalWinningTrades = trades.filter((t) => t.outcome === 'won').length;
    const totalLosingTrades = trades.filter((t) => t.outcome === 'lost').length;
    const totalPushTrades = trades.filter((t) => t.outcome === 'push').length;

    // Win rate calculation (excluding push from denominator is optional, we include all)
    const overallWinRate = totalTrades > 0 ? totalWinningTrades / totalTrades : 0;

    // Average ROI
    const totalReturn = trades.reduce((sum, t) => sum + t.returnPercent, 0);
    const averageRoiPerTrade = totalReturn / totalTrades;

    // Expectancy: pWin * avgWin - (1 - pWin) * avgLoss
    const expectancyPerTrade = this.computeExpectancy(trades);

    // Totals
    const totalStakeAmount = trades.reduce((sum, t) => sum + t.stakeAmount, 0);
    const totalProfitLoss = trades.reduce((sum, t) => sum + t.profitLoss, 0);

    return {
      totalTrades,
      totalWinningTrades,
      totalLosingTrades,
      totalPushTrades,
      overallWinRate: this.round(overallWinRate, 4),
      averageRoiPerTrade: this.round(averageRoiPerTrade, 4),
      expectancyPerTrade: this.round(expectancyPerTrade, 4),
      totalStakeAmount: this.round(totalStakeAmount, 2),
      totalProfitLoss: this.round(totalProfitLoss, 2),
    };
  }

  /**
   * Compute expectancy per trade
   * 
   * Formula: expectancy = pWin * avgWin - pLoss * avgLoss
   * Where:
   *   pWin = probability of winning
   *   avgWin = average return when winning (positive value)
   *   pLoss = probability of losing (1 - pWin - pPush)
   *   avgLoss = average loss when losing (positive value, absolute)
   */
  private computeExpectancy(trades: ResolvedTrade[]): number {
    const wins = trades.filter((t) => t.outcome === 'won');
    const losses = trades.filter((t) => t.outcome === 'lost');
    const totalTrades = trades.length;

    if (totalTrades === 0) return 0;

    const pWin = wins.length / totalTrades;
    const pLoss = losses.length / totalTrades;

    const avgWin = wins.length > 0
      ? wins.reduce((sum, t) => sum + t.returnPercent, 0) / wins.length
      : 0;

    const avgLoss = losses.length > 0
      ? Math.abs(losses.reduce((sum, t) => sum + t.returnPercent, 0) / losses.length)
      : 0;

    return pWin * avgWin - pLoss * avgLoss;
  }

  /**
   * Compute metrics grouped by category
   */
  private computeCategoryMetrics(trades: ResolvedTrade[]): CategoryStatsDto[] {
    // Group trades by category
    const categoryMap = new Map<string, ResolvedTrade[]>();

    for (const trade of trades) {
      const category = trade.category || 'other';
      if (!categoryMap.has(category)) {
        categoryMap.set(category, []);
      }
      categoryMap.get(category)!.push(trade);
    }

    // Compute metrics for each category
    const results: CategoryStatsDto[] = [];

    for (const [category, categoryTrades] of categoryMap) {
      const totalTrades = categoryTrades.length;
      const totalWinningTrades = categoryTrades.filter((t) => t.outcome === 'won').length;
      const totalLosingTrades = categoryTrades.filter((t) => t.outcome === 'lost').length;
      const totalPushTrades = categoryTrades.filter((t) => t.outcome === 'push').length;

      const winRate = totalTrades > 0 ? totalWinningTrades / totalTrades : 0;

      const totalReturn = categoryTrades.reduce((sum, t) => sum + t.returnPercent, 0);
      const averageRoiPerTrade = totalTrades > 0 ? totalReturn / totalTrades : 0;

      const totalStakeAmount = categoryTrades.reduce((sum, t) => sum + t.stakeAmount, 0);

      results.push({
        category,
        totalTrades,
        totalWinningTrades,
        totalLosingTrades,
        totalPushTrades,
        winRate: this.round(winRate, 4),
        averageRoiPerTrade: this.round(averageRoiPerTrade, 4),
        totalStakeAmount: this.round(totalStakeAmount, 2),
      });
    }

    // Sort by total trades descending
    results.sort((a, b) => b.totalTrades - a.totalTrades);

    return results;
  }

  /**
   * Compute equity curve and maximum drawdown
   * 
   * EQUITY CURVE LOGIC:
   * - Start with equity = 0
   * - For each trade (sorted by executedAt), add the trade's returnPercent to equity
   * - This gives cumulative return over time
   * 
   * MAX DRAWDOWN LOGIC:
   * - Track the peak equity seen so far
   * - At each point, calculate drawdown = (peak - current) / peak (if peak > 0)
   * - maxDrawdown = maximum drawdown seen
   * 
   * Note: We use additive returns (equity += return) rather than multiplicative
   * for simplicity. This represents "return on initial capital" over time.
   */
  private computeEquityCurveAndDrawdown(trades: ResolvedTrade[]): EquityCurveDto {
    if (trades.length === 0) {
      return {
        points: [],
        maxDrawdown: 0,
        finalEquity: 0,
        totalTrades: 0,
      };
    }

    // Sort by executedAt (should already be sorted, but ensure)
    const sortedTrades = [...trades].sort(
      (a, b) => a.executedAt.getTime() - b.executedAt.getTime(),
    );

    const points: EquityCurvePointDto[] = [];
    let equity = 0;
    let peak = 0;
    let maxDrawdown = 0;

    for (const trade of sortedTrades) {
      // Add this trade's return to cumulative equity
      equity += trade.returnPercent;

      // Track peak
      if (equity > peak) {
        peak = equity;
      }

      // Calculate current drawdown from peak
      // Drawdown is measured as absolute drop from peak
      // For percentage-based equity, drawdown = peak - current
      const currentDrawdown = peak - equity;
      if (currentDrawdown > maxDrawdown) {
        maxDrawdown = currentDrawdown;
      }

      points.push({
        executedAt: trade.executedAt.toISOString(),
        equity: this.round(equity, 4),
        tradeReturn: this.round(trade.returnPercent, 4),
        marketTitle: trade.marketTitle,
      });
    }

    return {
      points,
      maxDrawdown: this.round(maxDrawdown, 4),
      finalEquity: this.round(equity, 4),
      totalTrades: sortedTrades.length,
    };
  }

  // ============================================
  // UTILITY HELPERS
  // ============================================

  private round(value: number, decimals: number): number {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
  }
}
