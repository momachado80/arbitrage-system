import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { TradeDirection } from '../../../common/enums';

export class ExecutionSignalDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ enum: TradeDirection, example: TradeDirection.YES })
  direction: TradeDirection;

  @ApiProperty({ example: 75 })
  confidenceScore: number;

  @ApiProperty({ example: 0.45 })
  entryPriceSuggested: number;
}

export class ExecutionMarketDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: 'Trump wins 2028 election' })
  title: string;

  @ApiProperty({ example: 'politics' })
  category: string;
}

export class ExecutionResponseDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  signalId: string;

  @ApiProperty({ type: ExecutionSignalDto })
  signal: ExecutionSignalDto;

  @ApiPropertyOptional({ type: ExecutionMarketDto })
  market?: ExecutionMarketDto;

  @ApiProperty({ example: '2024-01-15T10:30:00.000Z' })
  executedAt: string;

  @ApiProperty({ example: 0.47 })
  entryPriceReal: number;

  @ApiProperty({ example: 100.0 })
  stakeAmount: number;

  @ApiPropertyOptional({ example: 'Executed via Polymarket web' })
  notes: string | null;
}
