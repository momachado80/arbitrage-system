import { PartialType, OmitType } from '@nestjs/swagger';
import { CreateSignalDto } from './create-signal.dto';

// Exclude marketId from updates - signals shouldn't change their market
export class UpdateSignalDto extends PartialType(
  OmitType(CreateSignalDto, ['marketId'] as const),
) {}
