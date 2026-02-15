import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsUUID,
  IsNotEmpty,
  IsDateString,
  IsNumber,
  IsOptional,
  IsString,
  Min,
} from 'class-validator';

export class CreateExecutionDto {
  @ApiProperty({
    description: 'UUID of the signal this execution is for',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @IsUUID()
  @IsNotEmpty()
  signalId: string;

  @ApiProperty({
    description: 'Timestamp when the trade was executed (ISO 8601)',
    example: '2024-01-15T10:30:00Z',
  })
  @IsDateString()
  executedAt: string;

  @ApiProperty({
    description: 'Real entry price at execution',
    example: 0.47,
  })
  @IsNumber()
  @Min(0)
  entryPriceReal: number;

  @ApiProperty({
    description: 'Stake amount in USD/USDC',
    example: 100.0,
  })
  @IsNumber()
  @Min(0)
  stakeAmount: number;

  @ApiPropertyOptional({
    description: 'Optional notes about the execution',
    example: 'Executed via Polymarket web interface',
  })
  @IsOptional()
  @IsString()
  notes?: string;
}
