import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { MarketOutcomeResult } from '../../../common/enums';

export class OutcomeMarketDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: 'Trump wins 2028 election' })
  title: string;

  @ApiProperty({ example: 'politics' })
  category: string;

  @ApiProperty({ example: 'polymarket' })
  source: string;
}

export class MarketOutcomeResponseDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  marketId: string;

  @ApiProperty({ type: OutcomeMarketDto })
  market: OutcomeMarketDto;

  @ApiProperty({ example: '2024-11-06T00:00:00.000Z' })
  resolvedAt: string;

  @ApiProperty({ enum: MarketOutcomeResult, example: MarketOutcomeResult.WON })
  outcome: MarketOutcomeResult;

  @ApiProperty({ example: 1.0 })
  finalPrice: number;

  @ApiPropertyOptional({ example: 'Election results confirmed' })
  notes: string | null;
}
