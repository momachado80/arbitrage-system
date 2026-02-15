import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { SignalsService } from './signals.service';
import { SignalsController } from './signals.controller';
import { Signal } from './entities/signal.entity';
import { MarketsModule } from '../markets/markets.module';

@Module({
  imports: [TypeOrmModule.forFeature([Signal]), MarketsModule],
  controllers: [SignalsController],
  providers: [SignalsService],
  exports: [SignalsService],
})
export class SignalsModule {}
