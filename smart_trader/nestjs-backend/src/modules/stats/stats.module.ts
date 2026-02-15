import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { StatsService } from './stats.service';
import { StatsController } from './stats.controller';
import { Execution } from '../executions/entities/execution.entity';
import { MarketOutcome } from '../market-outcomes/entities/market-outcome.entity';

@Module({
  imports: [
    TypeOrmModule.forFeature([Execution, MarketOutcome]),
  ],
  controllers: [StatsController],
  providers: [StatsService],
  exports: [StatsService],
})
export class StatsModule {}
