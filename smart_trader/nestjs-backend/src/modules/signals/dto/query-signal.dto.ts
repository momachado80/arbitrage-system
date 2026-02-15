import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsString, IsBoolean, IsEnum } from 'class-validator';
import { Transform } from 'class-transformer';
import { PaginationDto } from '../../../common/dto/pagination.dto';
import { SignalStatus } from '../../../common/enums';

export class QuerySignalDto extends PaginationDto {
  @ApiPropertyOptional({
    description: 'Filter by market category',
    example: 'politics',
  })
  @IsOptional()
  @IsString()
  category?: string;

  @ApiPropertyOptional({
    description: 'Filter by signal status',
    enum: SignalStatus,
  })
  @IsOptional()
  @IsEnum(SignalStatus)
  status?: SignalStatus;

  @ApiPropertyOptional({
    description: 'Filter urgent signals only',
    example: true,
  })
  @IsOptional()
  @Transform(({ value }) => value === 'true' || value === true)
  @IsBoolean()
  isUrgent?: boolean;
}
