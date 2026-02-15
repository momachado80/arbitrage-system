/**
 * Database Seeder for Smart Event Trader
 * 
 * This script creates sample data for testing the Stats endpoints.
 * Run with: npx ts-node src/database/seed.ts
 */

import { DataSource } from 'typeorm';
import { v4 as uuidv4 } from 'uuid';

// Initialize data source for SQLite
const dataSource = new DataSource({
  type: 'better-sqlite3',
  database: 'smart_event_trader.db',
  entities: [__dirname + '/../**/*.entity{.ts,.js}'],
  synchronize: true,
});

interface SeedMarket {
  id: string;
  title: string;
  source: string;
  externalId: string;
  category: string;
  resolutionDate: Date;
  status: string;
}

interface SeedSignal {
  id: string;
  marketId: string;
  direction: string;
  confidenceScore: number;
  entryPriceSuggested: number;
  targetPrice: number;
  stopPrice: number;
  rationale: string;
  isUrgent: boolean;
  status: string;
}

interface SeedExecution {
  id: string;
  signalId: string;
  executedAt: Date;
  entryPriceReal: number;
  stakeAmount: number;
  notes: string;
}

interface SeedOutcome {
  id: string;
  marketId: string;
  resolvedAt: Date;
  outcome: string;
  finalPrice: number;
  notes: string;
}

async function seed() {
  console.log('🌱 Starting database seed...\n');

  await dataSource.initialize();

  // Clear existing data
  await dataSource.query('DELETE FROM market_outcomes');
  await dataSource.query('DELETE FROM executions');
  await dataSource.query('DELETE FROM signals');
  await dataSource.query('DELETE FROM markets');

  console.log('✓ Cleared existing data\n');

  // Sample markets
  const markets: SeedMarket[] = [
    {
      id: uuidv4(),
      title: 'Trump wins 2028 Presidential Election',
      source: 'polymarket',
      externalId: 'trump-2028-pres',
      category: 'politics',
      resolutionDate: new Date('2028-11-06'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Bitcoin above $100k by end of 2025',
      source: 'polymarket',
      externalId: 'btc-100k-2025',
      category: 'crypto',
      resolutionDate: new Date('2025-12-31'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Fed cuts rates in March 2025',
      source: 'polymarket',
      externalId: 'fed-cut-mar-2025',
      category: 'finance',
      resolutionDate: new Date('2025-03-31'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Apple releases AR glasses in 2025',
      source: 'polymarket',
      externalId: 'apple-ar-2025',
      category: 'tech',
      resolutionDate: new Date('2025-12-31'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Democrats win Senate 2026',
      source: 'polymarket',
      externalId: 'dem-senate-2026',
      category: 'politics',
      resolutionDate: new Date('2026-11-04'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'ETH flips BTC market cap in 2025',
      source: 'polymarket',
      externalId: 'eth-flip-2025',
      category: 'crypto',
      resolutionDate: new Date('2025-12-31'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Lakers win NBA Championship 2025',
      source: 'polymarket',
      externalId: 'lakers-nba-2025',
      category: 'sports',
      resolutionDate: new Date('2025-06-20'),
      status: 'resolved',
    },
    {
      id: uuidv4(),
      title: 'Tesla stock above $400 by mid-2025',
      source: 'polymarket',
      externalId: 'tsla-400-2025',
      category: 'finance',
      resolutionDate: new Date('2025-06-30'),
      status: 'resolved',
    },
  ];

  // Insert markets
  for (const market of markets) {
    await dataSource.query(
      `INSERT INTO markets (id, title, source, external_id, category, resolution_date, status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))`,
      [market.id, market.title, market.source, market.externalId, market.category, market.resolutionDate.toISOString(), market.status]
    );
  }
  console.log(`✓ Created ${markets.length} markets\n`);

  // Create signals and executions for each market
  const signals: SeedSignal[] = [];
  const executions: SeedExecution[] = [];
  const outcomes: SeedOutcome[] = [];

  // Define outcomes for each market (mix of wins, losses, pushes)
  const marketOutcomes: { marketIndex: number; outcome: 'won' | 'lost' | 'push'; entryPrice: number; finalPrice: number }[] = [
    { marketIndex: 0, outcome: 'won', entryPrice: 0.45, finalPrice: 1.0 },   // Trump - won
    { marketIndex: 1, outcome: 'won', entryPrice: 0.35, finalPrice: 1.0 },   // BTC - won
    { marketIndex: 2, outcome: 'lost', entryPrice: 0.60, finalPrice: 0.0 },  // Fed - lost
    { marketIndex: 3, outcome: 'lost', entryPrice: 0.40, finalPrice: 0.0 },  // Apple AR - lost
    { marketIndex: 4, outcome: 'won', entryPrice: 0.55, finalPrice: 1.0 },   // Dem Senate - won
    { marketIndex: 5, outcome: 'lost', entryPrice: 0.25, finalPrice: 0.0 },  // ETH flip - lost
    { marketIndex: 6, outcome: 'push', entryPrice: 0.30, finalPrice: 0.30 }, // Lakers - push (cancelled)
    { marketIndex: 7, outcome: 'won', entryPrice: 0.50, finalPrice: 1.0 },   // Tesla - won
  ];

  let dayOffset = 0;
  for (const mo of marketOutcomes) {
    const market = markets[mo.marketIndex];
    const signalId = uuidv4();
    const executionId = uuidv4();
    const outcomeId = uuidv4();

    // Create signal
    signals.push({
      id: signalId,
      marketId: market.id,
      direction: 'YES',
      confidenceScore: 60 + Math.floor(Math.random() * 30),
      entryPriceSuggested: mo.entryPrice,
      targetPrice: 0.80,
      stopPrice: 0.20,
      rationale: `Analysis suggests favorable odds for ${market.title}`,
      isUrgent: Math.random() > 0.7,
      status: 'resolved',
    });

    // Create execution (staggered dates)
    const executionDate = new Date();
    executionDate.setDate(executionDate.getDate() - 30 + dayOffset);
    dayOffset += 3;

    executions.push({
      id: executionId,
      signalId: signalId,
      executedAt: executionDate,
      entryPriceReal: mo.entryPrice + (Math.random() * 0.04 - 0.02), // slight variance
      stakeAmount: 50 + Math.floor(Math.random() * 150),
      notes: 'Executed via API',
    });

    // Create outcome
    const resolvedDate = new Date(executionDate);
    resolvedDate.setDate(resolvedDate.getDate() + 5 + Math.floor(Math.random() * 10));

    outcomes.push({
      id: outcomeId,
      marketId: market.id,
      resolvedAt: resolvedDate,
      outcome: mo.outcome,
      finalPrice: mo.finalPrice,
      notes: `Market resolved as ${mo.outcome}`,
    });
  }

  // Add a few more trades to some markets (multiple executions per market)
  const extraTrades: { marketIndex: number; entryPrice: number }[] = [
    { marketIndex: 0, entryPrice: 0.42 },
    { marketIndex: 1, entryPrice: 0.38 },
    { marketIndex: 4, entryPrice: 0.52 },
  ];

  for (const et of extraTrades) {
    const market = markets[et.marketIndex];
    const existingOutcome = outcomes.find(o => o.marketId === market.id);
    if (!existingOutcome) continue;

    const signalId = uuidv4();
    const executionId = uuidv4();

    signals.push({
      id: signalId,
      marketId: market.id,
      direction: 'YES',
      confidenceScore: 65 + Math.floor(Math.random() * 25),
      entryPriceSuggested: et.entryPrice,
      targetPrice: 0.75,
      stopPrice: 0.25,
      rationale: `Additional position on ${market.title}`,
      isUrgent: false,
      status: 'resolved',
    });

    const executionDate = new Date();
    executionDate.setDate(executionDate.getDate() - 25 + dayOffset);
    dayOffset += 2;

    executions.push({
      id: executionId,
      signalId: signalId,
      executedAt: executionDate,
      entryPriceReal: et.entryPrice + (Math.random() * 0.02 - 0.01),
      stakeAmount: 75 + Math.floor(Math.random() * 100),
      notes: 'Additional position',
    });
  }

  // Insert signals
  for (const signal of signals) {
    await dataSource.query(
      `INSERT INTO signals (id, market_id, direction, confidence_score, entry_price_suggested, target_price, stop_price, rationale, is_urgent, status, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))`,
      [signal.id, signal.marketId, signal.direction, signal.confidenceScore, signal.entryPriceSuggested, signal.targetPrice, signal.stopPrice, signal.rationale, signal.isUrgent ? 1 : 0, signal.status]
    );
  }
  console.log(`✓ Created ${signals.length} signals\n`);

  // Insert executions
  for (const execution of executions) {
    await dataSource.query(
      `INSERT INTO executions (id, signal_id, executed_at, entry_price_real, stake_amount, notes)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [execution.id, execution.signalId, execution.executedAt.toISOString(), execution.entryPriceReal, execution.stakeAmount, execution.notes]
    );
  }
  console.log(`✓ Created ${executions.length} executions\n`);

  // Insert outcomes
  for (const outcome of outcomes) {
    await dataSource.query(
      `INSERT INTO market_outcomes (id, market_id, resolved_at, outcome, final_price, notes)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [outcome.id, outcome.marketId, outcome.resolvedAt.toISOString(), outcome.outcome, outcome.finalPrice, outcome.notes]
    );
  }
  console.log(`✓ Created ${outcomes.length} market outcomes\n`);

  await dataSource.destroy();

  console.log('═══════════════════════════════════════════════════════════');
  console.log('✅ Seed completed successfully!');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('\nSummary:');
  console.log(`  - ${markets.length} markets`);
  console.log(`  - ${signals.length} signals`);
  console.log(`  - ${executions.length} executions (trades)`);
  console.log(`  - ${outcomes.length} market outcomes`);
  console.log('\nTest the stats endpoints:');
  console.log('  GET http://localhost:3000/stats/overview');
  console.log('  GET http://localhost:3000/stats/by-category');
  console.log('  GET http://localhost:3000/stats/equity-curve');
  console.log('');
}

seed().catch((err) => {
  console.error('Seed failed:', err);
  process.exit(1);
});
