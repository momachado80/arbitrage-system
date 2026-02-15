import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Market } from '../markets/entities/market.entity';
import { Signal } from '../signals/entities/signal.entity';
import { Execution } from '../executions/entities/execution.entity';
import { MarketOutcome } from '../market-outcomes/entities/market-outcome.entity';
import {
  MarketStatus,
  SignalStatus,
  TradeDirection,
  MarketOutcomeResult,
} from '../../common/enums';

/**
 * ⚠️  DEV SERVICE - DO NOT ENABLE IN PRODUCTION ⚠️
 * 
 * This service provides database seeding functionality for development/testing.
 * It will DELETE all existing data before inserting demo data.
 */
@Injectable()
export class DevService {
  constructor(
    @InjectRepository(Market)
    private readonly marketRepository: Repository<Market>,
    @InjectRepository(Signal)
    private readonly signalRepository: Repository<Signal>,
    @InjectRepository(Execution)
    private readonly executionRepository: Repository<Execution>,
    @InjectRepository(MarketOutcome)
    private readonly marketOutcomeRepository: Repository<MarketOutcome>,
  ) {}

  /**
   * Seeds the database with demo data.
   * 
   * ⚠️ WARNING: This method DELETES all existing data before inserting!
   * 
   * Insertion order (respects foreign keys):
   * 1. Markets
   * 2. Signals (linked to markets)
   * 3. Executions (linked to signals)
   * 4. MarketOutcomes (linked to markets)
   */
  async seed(): Promise<{
    markets: number;
    signals: number;
    executions: number;
    outcomes: number;
  }> {
    // ========================================
    // STEP 1: Clear existing data
    // ========================================
    // Delete in reverse order of dependencies
    await this.executionRepository.delete({});
    await this.marketOutcomeRepository.delete({});
    await this.signalRepository.delete({});
    await this.marketRepository.delete({});

    // ========================================
    // STEP 2: Create Markets
    // ========================================
    const marketsData: Partial<Market>[] = [
      {
        title: 'Trump wins 2028 US Presidential Election',
        source: 'polymarket',
        externalId: 'trump-2028-president',
        category: 'politics',
        resolutionDate: new Date('2028-11-06'),
        status: MarketStatus.RESOLVED,
      },
      {
        title: 'Bitcoin above $100k by end of 2027',
        source: 'polymarket',
        externalId: 'btc-100k-2027',
        category: 'crypto',
        resolutionDate: new Date('2027-12-31'),
        status: MarketStatus.RESOLVED,
      },
      {
        title: 'Apple stock up 10% this quarter',
        source: 'polymarket',
        externalId: 'aapl-up-10pct-q1',
        category: 'tech',
        resolutionDate: new Date('2026-03-31'),
        status: MarketStatus.RESOLVED,
      },
      {
        title: 'Real Madrid wins Champions League 2026',
        source: 'polymarket',
        externalId: 'real-madrid-ucl-2026',
        category: 'sports',
        resolutionDate: new Date('2026-06-01'),
        status: MarketStatus.RESOLVED,
      },
      {
        title: 'Fed cuts rates in Q2 2026',
        source: 'polymarket',
        externalId: 'fed-rate-cut-q2-2026',
        category: 'finance',
        resolutionDate: new Date('2026-06-30'),
        status: MarketStatus.OPEN, // Still open - no outcome yet
      },
    ];

    const markets = await this.marketRepository.save(
      marketsData.map((m) => this.marketRepository.create(m)),
    );

    // ========================================
    // STEP 3: Create Signals for each market
    // ========================================
    const signalsData: Partial<Signal>[] = [
      // Trump 2028 - 2 signals (1 active, 1 resolved)
      {
        marketId: markets[0].id,
        direction: TradeDirection.YES,
        confidenceScore: 75,
        entryPriceSuggested: 0.45,
        targetPrice: 0.70,
        stopPrice: 0.30,
        rationale: 'Strong polling data and incumbent advantage analysis',
        isUrgent: true,
        status: SignalStatus.ACTIVE,
      },
      {
        marketId: markets[0].id,
        direction: TradeDirection.YES,
        confidenceScore: 68,
        entryPriceSuggested: 0.52,
        targetPrice: 0.75,
        stopPrice: 0.35,
        rationale: 'Additional position after momentum shift',
        isUrgent: false,
        status: SignalStatus.RESOLVED,
      },
      // Bitcoin 100k - 1 active signal
      {
        marketId: markets[1].id,
        direction: TradeDirection.YES,
        confidenceScore: 82,
        entryPriceSuggested: 0.35,
        targetPrice: 0.60,
        stopPrice: 0.20,
        rationale: 'Halving cycle analysis + institutional adoption trends',
        isUrgent: true,
        status: SignalStatus.ACTIVE,
      },
      // Apple stock - 1 active signal
      {
        marketId: markets[2].id,
        direction: TradeDirection.YES,
        confidenceScore: 60,
        entryPriceSuggested: 0.55,
        targetPrice: 0.75,
        stopPrice: 0.40,
        rationale: 'Product launch catalyst expected',
        isUrgent: false,
        status: SignalStatus.ACTIVE,
      },
      // Real Madrid UCL - 2 signals (1 active, 1 resolved)
      {
        marketId: markets[3].id,
        direction: TradeDirection.YES,
        confidenceScore: 70,
        entryPriceSuggested: 0.30,
        targetPrice: 0.55,
        stopPrice: 0.18,
        rationale: 'Strong squad depth and favorable draw',
        isUrgent: false,
        status: SignalStatus.ACTIVE,
      },
      {
        marketId: markets[3].id,
        direction: TradeDirection.NO,
        confidenceScore: 65,
        entryPriceSuggested: 0.70,
        targetPrice: 0.85,
        stopPrice: 0.55,
        rationale: 'Hedge position - injury concerns',
        isUrgent: false,
        status: SignalStatus.RESOLVED,
      },
      // Fed rate cut - 1 active signal
      {
        marketId: markets[4].id,
        direction: TradeDirection.YES,
        confidenceScore: 72,
        entryPriceSuggested: 0.48,
        targetPrice: 0.70,
        stopPrice: 0.30,
        rationale: 'Economic indicators pointing to rate cut',
        isUrgent: true,
        status: SignalStatus.ACTIVE,
      },
    ];

    const signals = await this.signalRepository.save(
      signalsData.map((s) => this.signalRepository.create(s)),
    );

    // ========================================
    // STEP 4: Create Executions for some signals
    // ========================================
    const now = new Date();
    const daysAgo = (days: number) => {
      const date = new Date(now);
      date.setDate(date.getDate() - days);
      return date;
    };

    const executionsData: Partial<Execution>[] = [
      // Trump signal 1
      {
        signalId: signals[0].id,
        executedAt: daysAgo(30),
        entryPriceReal: 0.46,
        stakeAmount: 200,
        notes: 'Initial position',
      },
      // Trump signal 2
      {
        signalId: signals[1].id,
        executedAt: daysAgo(20),
        entryPriceReal: 0.53,
        stakeAmount: 150,
        notes: 'Added to position',
      },
      // Bitcoin signal
      {
        signalId: signals[2].id,
        executedAt: daysAgo(25),
        entryPriceReal: 0.36,
        stakeAmount: 500,
        notes: 'High conviction trade',
      },
      // Apple signal
      {
        signalId: signals[3].id,
        executedAt: daysAgo(15),
        entryPriceReal: 0.54,
        stakeAmount: 100,
        notes: 'Small position',
      },
      // Real Madrid YES signal
      {
        signalId: signals[4].id,
        executedAt: daysAgo(40),
        entryPriceReal: 0.31,
        stakeAmount: 250,
        notes: 'Pre-tournament bet',
      },
      // Real Madrid NO signal (hedge)
      {
        signalId: signals[5].id,
        executedAt: daysAgo(35),
        entryPriceReal: 0.68,
        stakeAmount: 100,
        notes: 'Hedge position',
      },
      // Fed rate cut (still open)
      {
        signalId: signals[6].id,
        executedAt: daysAgo(5),
        entryPriceReal: 0.49,
        stakeAmount: 300,
        notes: 'Macro play',
      },
    ];

    const executions = await this.executionRepository.save(
      executionsData.map((e) => this.executionRepository.create(e)),
    );

    // ========================================
    // STEP 5: Create MarketOutcomes for resolved markets
    // ========================================
    const outcomesData: Partial<MarketOutcome>[] = [
      // Trump 2028 - WON
      {
        marketId: markets[0].id,
        resolvedAt: daysAgo(2),
        outcome: MarketOutcomeResult.WON,
        finalPrice: 1.0,
        notes: 'Trump won the 2028 election',
      },
      // Bitcoin 100k - WON
      {
        marketId: markets[1].id,
        resolvedAt: daysAgo(5),
        outcome: MarketOutcomeResult.WON,
        finalPrice: 1.0,
        notes: 'Bitcoin reached $100k in December 2027',
      },
      // Apple stock - LOST
      {
        marketId: markets[2].id,
        resolvedAt: daysAgo(10),
        outcome: MarketOutcomeResult.LOST,
        finalPrice: 0.0,
        notes: 'Apple stock only rose 6% - below 10% threshold',
      },
      // Real Madrid UCL - PUSH (tournament cancelled due to unforeseen event)
      {
        marketId: markets[3].id,
        resolvedAt: daysAgo(8),
        outcome: MarketOutcomeResult.PUSH,
        finalPrice: 0.5, // Stake returned
        notes: 'Market voided - stakes returned',
      },
      // Fed rate cut - NO OUTCOME (market still open)
    ];

    const outcomes = await this.marketOutcomeRepository.save(
      outcomesData.map((o) => this.marketOutcomeRepository.create(o)),
    );

    return {
      markets: markets.length,
      signals: signals.length,
      executions: executions.length,
      outcomes: outcomes.length,
    };
  }

  /**
   * Returns current counts for all tables (useful for verification)
   */
  async getCounts(): Promise<{
    markets: number;
    signals: number;
    executions: number;
    outcomes: number;
  }> {
    const [markets, signals, executions, outcomes] = await Promise.all([
      this.marketRepository.count(),
      this.signalRepository.count(),
      this.executionRepository.count(),
      this.marketOutcomeRepository.count(),
    ]);

    return { markets, signals, executions, outcomes };
  }
}
