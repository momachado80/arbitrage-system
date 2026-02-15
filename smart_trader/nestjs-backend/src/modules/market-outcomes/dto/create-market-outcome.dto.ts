import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsUUID,
  IsNotEmpty,
  IsDateString,
  IsEnum,
  IsNumber,
  IsOptional,
  IsString,
  Min,
  Max,
} from 'class-validator';
import { MarketOutcomeResult } from '../../../common/enums';

export class CreateMarketOutcomeDto {
  @ApiProperty({
    description: 'UUID of the market this outcome is for',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @IsUUID()
  @IsNotEmpty()
  marketId: string;

  @ApiProperty({
    description: 'Timestamp when the market was resolved (ISO 8601)',
    example: '2024-11-06T00:00:00Z',
  })
  @IsDateString()
  resolvedAt: string;

  @ApiProperty({
    description: 'Outcome of the market resolution',
    enum: MarketOutcomeResult,
    example: MarketOutcomeResult.WON,
  })
  @IsEnum(MarketOutcomeResult)
  outcome: MarketOutcomeResult;

  @ApiProperty({
    description: 'Final settlement price/probability (0-1 for binary markets)',
    example: 1.0,
  })
  @IsNumber()
  @Min(0)
  @Max(1)
  finalPrice: number;

  @ApiPropertyOptional({
    description: 'Optional notes about the resolution',
    example: 'Market resolved YES after election results confirmed',
  })
  @IsOptional()
  @IsString()
  notes?: string;
}
