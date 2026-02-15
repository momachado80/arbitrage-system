import { Controller, Get, Query } from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiResponse,
  ApiQuery,
} from '@nestjs/swagger';
import { StatsService } from './stats.service';
import {
  StatsQueryDto,
  OverviewStatsDto,
  CategoryStatsDto,
  EquityCurveDto,
} from './dto/stats.dto';

@ApiTags('stats')
@Controller('stats')
export class StatsController {
  constructor(private readonly statsService: StatsService) {}

  @Get('overview')
  @ApiOperation({
    summary: 'Get overall trading statistics',
    description: `
      Returns aggregated statistics for all resolved trades:
      - Total trades (wins, losses, pushes)
      - Overall win rate
      - Average ROI per trade
      - Expectancy per trade
      - Total stake and profit/loss
      
      Only includes trades with a market outcome (open trades excluded).
    `,
  })
  @ApiQuery({
    name: 'fromDate',
    required: false,
    description: 'Filter from date (ISO 8601)',
    example: '2024-01-01T00:00:00Z',
  })
  @ApiQuery({
    name: 'toDate',
    required: false,
    description: 'Filter to date (ISO 8601)',
    example: '2024-12-31T23:59:59Z',
  })
  @ApiResponse({
    status: 200,
    description: 'Overview statistics returned successfully',
    type: OverviewStatsDto,
  })
  getOverviewStats(@Query() query: StatsQueryDto): Promise<OverviewStatsDto> {
    return this.statsService.getOverviewStats(query);
  }

  @Get('by-category')
  @ApiOperation({
    summary: 'Get statistics grouped by market category',
    description: `
      Returns trading statistics broken down by category (politics, crypto, sports, etc.):
      - Total trades per category
      - Win rate per category
      - Average ROI per category
      
      Useful for identifying which categories perform best.
    `,
  })
  @ApiQuery({
    name: 'fromDate',
    required: false,
    description: 'Filter from date (ISO 8601)',
  })
  @ApiQuery({
    name: 'toDate',
    required: false,
    description: 'Filter to date (ISO 8601)',
  })
  @ApiResponse({
    status: 200,
    description: 'Category statistics returned successfully',
    type: [CategoryStatsDto],
  })
  getStatsByCategory(@Query() query: StatsQueryDto): Promise<CategoryStatsDto[]> {
    return this.statsService.getStatsByCategory(query);
  }

  @Get('equity-curve')
  @ApiOperation({
    summary: 'Get equity curve and max drawdown',
    description: `
      Returns the equity curve (cumulative returns over time) and maximum drawdown:
      - Points: array of {executedAt, equity, tradeReturn} in chronological order
      - Max drawdown: the largest peak-to-trough decline
      - Final equity: cumulative return at the end
      
      Equity starts at 0 and each trade adds its return percentage.
      Useful for visualizing trading performance over time.
    `,
  })
  @ApiQuery({
    name: 'fromDate',
    required: false,
    description: 'Filter from date (ISO 8601)',
  })
  @ApiQuery({
    name: 'toDate',
    required: false,
    description: 'Filter to date (ISO 8601)',
  })
  @ApiResponse({
    status: 200,
    description: 'Equity curve returned successfully',
    type: EquityCurveDto,
  })
  getEquityCurve(@Query() query: StatsQueryDto): Promise<EquityCurveDto> {
    return this.statsService.getEquityCurve(query);
  }
}
