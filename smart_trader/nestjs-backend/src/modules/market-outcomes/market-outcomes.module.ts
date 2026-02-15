import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MarketOutcomesService } from './market-outcomes.service';
import { MarketOutcomesController } from './market-outcomes.controller';
import { MarketOutcome } from './entities/market-outcome.entity';
import { MarketsModule } from '../markets/markets.module';

@Module({
  imports: [TypeOrmModule.forFeature([MarketOutcome]), MarketsModule],
  controllers: [MarketOutcomesController],
  providers: [MarketOutcomesService],
  exports: [MarketOutcomesService],
})
export class MarketOutcomesModule {}
