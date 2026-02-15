import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { MarketStatus } from '../../../common/enums';

export class MarketResponseDto {
  @ApiProperty({ example: '550e8400-e29b-41d4-a716-446655440000' })
  id: string;

  @ApiProperty({ example: 'Trump wins 2028 election' })
  title: string;

  @ApiProperty({ example: 'polymarket' })
  source: string;

  @ApiProperty({ example: '0x1234567890abcdef' })
  externalId: string;

  @ApiProperty({ example: 'politics' })
  category: string;

  @ApiPropertyOptional({ example: '2028-11-05T00:00:00.000Z' })
  resolutionDate: string | null;

  @ApiProperty({ enum: MarketStatus, example: MarketStatus.OPEN })
  status: MarketStatus;

  @ApiProperty({ example: '2024-01-15T10:30:00.000Z' })
  createdAt: string;

  @ApiProperty({ example: '2024-01-15T10:30:00.000Z' })
  updatedAt: string;
}
