import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DevController } from './dev.controller';
import { DevService } from './dev.service';
import { Market } from '../markets/entities/market.entity';
import { Signal } from '../signals/entities/signal.entity';
import { Execution } from '../executions/entities/execution.entity';
import { MarketOutcome } from '../market-outcomes/entities/market-outcome.entity';

/**
 * ⚠️  DEV MODULE - DO NOT ENABLE IN PRODUCTION ⚠️
 * 
 * This module provides database seeding and utility endpoints for development.
 * 
 * To disable in production, either:
 * 1. Remove this module from AppModule imports
 * 2. Add environment-based conditional import
 * 3. Use guards to block access
 */
@Module({
  imports: [
    TypeOrmModule.forFeature([Market, Signal, Execution, MarketOutcome]),
  ],
  controllers: [DevController],
  providers: [DevService],
})
export class DevModule {}
