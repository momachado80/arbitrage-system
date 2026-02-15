import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsDateString } from 'class-validator';

// ============================================
// Query DTOs
// ============================================

export class StatsQueryDto {
  @ApiPropertyOptional({
    description: 'Filter stats from this date (ISO 8601)',
    example: '2024-01-01T00:00:00Z',
  })
  @IsOptional()
  @IsDateString()
  fromDate?: string;

  @ApiPropertyOptional({
    description: 'Filter stats until this date (ISO 8601)',
    example: '2024-12-31T23:59:59Z',
  })
  @IsOptional()
  @IsDateString()
  toDate?: string;
}

// ============================================
// Response DTOs
// ============================================

export class OverviewStatsDto {
  @ApiProperty({ description: 'Total number of completed trades', example: 150 })
  totalTrades: number;

  @ApiProperty({ description: 'Number of winning trades', example: 95 })
  totalWinningTrades: number;

  @ApiProperty({ description: 'Number of losing trades', example: 50 })
  totalLosingTrades: number;

  @ApiProperty({ description: 'Number of push/break-even trades', example: 5 })
  totalPushTrades: number;

  @ApiProperty({
    description: 'Overall win rate (0-1 scale, e.g., 0.63 = 63%)',
    example: 0.633,
  })
  overallWinRate: number;

  @ApiProperty({
    description: 'Average ROI per trade (e.g., 0.12 = 12%)',
    example: 0.12,
  })
  averageRoiPerTrade: number;

  @ApiPropertyOptional({
    description: 'Expectancy per trade: pWin * avgWin - (1 - pWin) * avgLoss',
    example: 0.08,
  })
  expectancyPerTrade: number;

  @ApiProperty({
    description: 'Total stake amount across all trades',
    example: 15000.0,
  })
  totalStakeAmount: number;

  @ApiProperty({
    description: 'Total profit/loss in absolute terms',
    example: 1800.0,
  })
  totalProfitLoss: number;
}

export class CategoryStatsDto {
  @ApiProperty({ description: 'Market category', example: 'politics' })
  category: string;

  @ApiProperty({ description: 'Total trades in this category', example: 45 })
  totalTrades: number;

  @ApiProperty({ description: 'Winning trades in this category', example: 30 })
  totalWinningTrades: number;

  @ApiProperty({ description: 'Losing trades in this category', example: 12 })
  totalLosingTrades: number;

  @ApiProperty({ description: 'Push trades in this category', example: 3 })
  totalPushTrades: number;

  @ApiProperty({ description: 'Win rate for this category (0-1)', example: 0.667 })
  winRate: number;

  @ApiProperty({ description: 'Average ROI for this category', example: 0.15 })
  averageRoiPerTrade: number;

  @ApiProperty({ description: 'Total stake in this category', example: 4500.0 })
  totalStakeAmount: number;
}

export class EquityCurvePointDto {
  @ApiProperty({
    description: 'Timestamp of the trade execution',
    example: '2024-01-15T10:30:00.000Z',
  })
  executedAt: string;

  @ApiProperty({
    description: 'Cumulative equity at this point (starting from 0)',
    example: 0.24,
  })
  equity: number;

  @ApiProperty({
    description: 'Return of this individual trade',
    example: 0.08,
  })
  tradeReturn: number;

  @ApiPropertyOptional({
    description: 'Market title for context',
    example: 'Trump wins 2028',
  })
  marketTitle?: string;
}

export class EquityCurveDto {
  @ApiProperty({
    description: 'Array of equity curve points in chronological order',
    type: [EquityCurvePointDto],
  })
  points: EquityCurvePointDto[];

  @ApiProperty({
    description: 'Maximum drawdown (peak-to-trough) as decimal (e.g., 0.15 = 15%)',
    example: 0.15,
  })
  maxDrawdown: number;

  @ApiProperty({
    description: 'Final cumulative equity',
    example: 0.45,
  })
  finalEquity: number;

  @ApiProperty({
    description: 'Total number of trades in the curve',
    example: 50,
  })
  totalTrades: number;
}

// ============================================
// Internal Types (for calculations)
// ============================================

/**
 * Internal type representing a resolved trade with all data needed for stats
 */
export interface ResolvedTrade {
  executionId: string;
  signalId: string;
  marketId: string;
  marketTitle: string;
  category: string;
  executedAt: Date;
  entryPriceReal: number;
  stakeAmount: number;
  outcome: 'won' | 'lost' | 'push';
  finalPrice: number;
  returnPercent: number;
  profitLoss: number;
}
