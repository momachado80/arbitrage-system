import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsString,
  IsNotEmpty,
  IsOptional,
  IsEnum,
  IsDateString,
  MaxLength,
} from 'class-validator';
import { MarketStatus, MarketCategory } from '../../../common/enums';

export class CreateMarketDto {
  @ApiProperty({
    description: 'Short name/title of the market event',
    example: 'Trump wins 2028 election',
    maxLength: 500,
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(500)
  title: string;

  @ApiProperty({
    description: 'Source platform of the market',
    example: 'polymarket',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(100)
  source: string;

  @ApiProperty({
    description: 'External ID from the source platform',
    example: '0x1234567890abcdef',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(255)
  externalId: string;

  @ApiPropertyOptional({
    description: 'Market category',
    enum: MarketCategory,
    example: MarketCategory.POLITICS,
  })
  @IsOptional()
  @IsString()
  category?: string = MarketCategory.OTHER;

  @ApiPropertyOptional({
    description: 'Expected resolution date (ISO 8601)',
    example: '2028-11-05T00:00:00Z',
  })
  @IsOptional()
  @IsDateString()
  resolutionDate?: string;

  @ApiPropertyOptional({
    description: 'Market status',
    enum: MarketStatus,
    default: MarketStatus.OPEN,
  })
  @IsOptional()
  @IsEnum(MarketStatus)
  status?: MarketStatus = MarketStatus.OPEN;
}
