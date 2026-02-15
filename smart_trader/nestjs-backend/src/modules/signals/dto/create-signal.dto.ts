import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsUUID,
  IsString,
  IsNotEmpty,
  IsEnum,
  IsInt,
  IsNumber,
  IsBoolean,
  IsOptional,
  Min,
  Max,
} from 'class-validator';
import { TradeDirection, SignalStatus } from '../../../common/enums';

export class CreateSignalDto {
  @ApiProperty({
    description: 'UUID of the market this signal is for',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @IsUUID()
  @IsNotEmpty()
  marketId: string;

  @ApiProperty({
    description: 'Trade direction',
    enum: TradeDirection,
    example: TradeDirection.YES,
  })
  @IsEnum(TradeDirection)
  direction: TradeDirection;

  @ApiProperty({
    description: 'Confidence score (0-100)',
    example: 75,
    minimum: 0,
    maximum: 100,
  })
  @IsInt()
  @Min(0)
  @Max(100)
  confidenceScore: number;

  @ApiProperty({
    description: 'Suggested entry price/probability',
    example: 0.45,
  })
  @IsNumber()
  @Min(0)
  entryPriceSuggested: number;

  @ApiProperty({
    description: 'Target/take-profit price',
    example: 0.65,
  })
  @IsNumber()
  @Min(0)
  targetPrice: number;

  @ApiProperty({
    description: 'Stop-loss price',
    example: 0.35,
  })
  @IsNumber()
  @Min(0)
  stopPrice: number;

  @ApiPropertyOptional({
    description: 'Rationale/explanation for the signal',
    example: 'Strong momentum + whale accumulation detected',
  })
  @IsOptional()
  @IsString()
  rationale?: string;

  @ApiPropertyOptional({
    description: 'Is this signal urgent?',
    example: true,
    default: false,
  })
  @IsOptional()
  @IsBoolean()
  isUrgent?: boolean = false;

  @ApiPropertyOptional({
    description: 'Signal status',
    enum: SignalStatus,
    default: SignalStatus.ACTIVE,
  })
  @IsOptional()
  @IsEnum(SignalStatus)
  status?: SignalStatus = SignalStatus.ACTIVE;
}
