import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsString, IsDateString, IsEnum } from 'class-validator';
import { PaginationDto } from '../../../common/dto/pagination.dto';
import { MarketStatus } from '../../../common/enums';

export class QueryMarketDto extends PaginationDto {
  @ApiPropertyOptional({
    description: 'Filter by category',
    example: 'politics',
  })
  @IsOptional()
  @IsString()
  category?: string;

  @ApiPropertyOptional({
    description: 'Filter by status',
    enum: MarketStatus,
  })
  @IsOptional()
  @IsEnum(MarketStatus)
  status?: MarketStatus;

  @ApiPropertyOptional({
    description: 'Filter markets resolving before this date (ISO 8601)',
    example: '2028-12-31T23:59:59Z',
  })
  @IsOptional()
  @IsDateString()
  resolutionBefore?: string;

  @ApiPropertyOptional({
    description: 'Filter markets resolving after this date (ISO 8601)',
    example: '2028-01-01T00:00:00Z',
  })
  @IsOptional()
  @IsDateString()
  resolutionAfter?: string;
}
