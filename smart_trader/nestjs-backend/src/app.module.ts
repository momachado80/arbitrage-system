import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MarketsModule } from './modules/markets/markets.module';
import { SignalsModule } from './modules/signals/signals.module';
import { ExecutionsModule } from './modules/executions/executions.module';
import { MarketOutcomesModule } from './modules/market-outcomes/market-outcomes.module';
import { StatsModule } from './modules/stats/stats.module';
import { DevModule } from './modules/dev/dev.module';

@Module({
  imports: [
    // Load environment variables
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),

    // TypeORM configuration - SQLite for simplicity
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'better-sqlite3',
        database: 'smart_event_trader.db',
        entities: [__dirname + '/**/*.entity{.ts,.js}'],
        synchronize: true, // Auto-create tables in dev
        logging: configService.get('NODE_ENV') === 'development',
      }),
      inject: [ConfigService],
    }),

    // Feature modules
    MarketsModule,
    SignalsModule,
    ExecutionsModule,
    MarketOutcomesModule,
    StatsModule,

    // ⚠️ DEV MODULE - Remove in production!
    DevModule,
  ],
})
export class AppModule {}
