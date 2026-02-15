import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { SignalStatus, TradeDirection } from '../../../common/enums';

export class SignalMarketDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: 'Trump wins 2028 election' })
  title: string;

  @ApiProperty({ example: 'politics' })
  category: string;

  @ApiProperty({ example: 'polymarket' })
  source: string;
}

export class SignalResponseDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  marketId: string;

  @ApiProperty({ type: SignalMarketDto })
  market: SignalMarketDto;

  @ApiProperty({ example: '2024-01-15T10:30:00.000Z' })
  createdAt: string;

  @ApiProperty({ enum: TradeDirection, example: TradeDirection.YES })
  direction: TradeDirection;

  @ApiProperty({ example: 75 })
  confidenceScore: number;

  @ApiProperty({ example: 0.45 })
  entryPriceSuggested: number;

  @ApiProperty({ example: 0.65 })
  targetPrice: number;

  @ApiProperty({ example: 0.35 })
  stopPrice: number;

  @ApiPropertyOptional({ example: 'Strong momentum + whale accumulation' })
  rationale: string | null;

  @ApiProperty({ example: true })
  isUrgent: boolean;

  @ApiProperty({ enum: SignalStatus, example: SignalStatus.ACTIVE })
  status: SignalStatus;
}
