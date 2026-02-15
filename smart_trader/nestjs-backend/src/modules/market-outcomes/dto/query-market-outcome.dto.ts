import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsUUID, IsEnum } from 'class-validator';
import { PaginationDto } from '../../../common/dto/pagination.dto';
import { MarketOutcomeResult } from '../../../common/enums';

export class QueryMarketOutcomeDto extends PaginationDto {
  @ApiPropertyOptional({
    description: 'Filter by market ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @IsOptional()
  @IsUUID()
  marketId?: string;

  @ApiPropertyOptional({
    description: 'Filter by outcome result',
    enum: MarketOutcomeResult,
  })
  @IsOptional()
  @IsEnum(MarketOutcomeResult)
  outcome?: MarketOutcomeResult;
}
