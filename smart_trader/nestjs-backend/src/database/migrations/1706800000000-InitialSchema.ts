import { MigrationInterface, QueryRunner } from 'typeorm';

export class InitialSchema1706800000000 implements MigrationInterface {
  name = 'InitialSchema1706800000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // Create markets table
    await queryRunner.query(`
      CREATE TABLE "markets" (
        "id" uuid NOT NULL DEFAULT uuid_generate_v4(),
        "title" varchar(500) NOT NULL,
        "source" varchar(100) NOT NULL,
        "external_id" varchar(255) NOT NULL,
        "category" varchar(50) NOT NULL DEFAULT 'other',
        "resolution_date" TIMESTAMP WITH TIME ZONE,
        "status" varchar(20) NOT NULL DEFAULT 'open',
        "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT "PK_markets" PRIMARY KEY ("id")
      )
    `);

    // Create signals table
    await queryRunner.query(`
      CREATE TABLE "signals" (
        "id" uuid NOT NULL DEFAULT uuid_generate_v4(),
        "market_id" uuid NOT NULL,
        "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        "direction" varchar(20) NOT NULL,
        "confidence_score" integer NOT NULL,
        "entry_price_suggested" numeric(10,4) NOT NULL,
        "target_price" numeric(10,4) NOT NULL,
        "stop_price" numeric(10,4) NOT NULL,
        "rationale" text,
        "is_urgent" boolean NOT NULL DEFAULT false,
        "status" varchar(20) NOT NULL DEFAULT 'active',
        CONSTRAINT "PK_signals" PRIMARY KEY ("id")
      )
    `);

    // Create executions table
    await queryRunner.query(`
      CREATE TABLE "executions" (
        "id" uuid NOT NULL DEFAULT uuid_generate_v4(),
        "signal_id" uuid NOT NULL,
        "executed_at" TIMESTAMP WITH TIME ZONE NOT NULL,
        "entry_price_real" numeric(10,4) NOT NULL,
        "stake_amount" numeric(18,2) NOT NULL,
        "notes" text,
        CONSTRAINT "PK_executions" PRIMARY KEY ("id")
      )
    `);

    // Create market_outcomes table
    await queryRunner.query(`
      CREATE TABLE "market_outcomes" (
        "id" uuid NOT NULL DEFAULT uuid_generate_v4(),
        "market_id" uuid NOT NULL,
        "resolved_at" TIMESTAMP WITH TIME ZONE NOT NULL,
        "outcome" varchar(20) NOT NULL,
        "final_price" numeric(10,4) NOT NULL,
        "notes" text,
        CONSTRAINT "PK_market_outcomes" PRIMARY KEY ("id")
      )
    `);

    // Enable uuid-ossp extension for uuid_generate_v4()
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`);

    // Add foreign key constraints
    await queryRunner.query(`
      ALTER TABLE "signals"
      ADD CONSTRAINT "FK_signals_market"
      FOREIGN KEY ("market_id") REFERENCES "markets"("id")
      ON DELETE CASCADE ON UPDATE NO ACTION
    `);

    await queryRunner.query(`
      ALTER TABLE "executions"
      ADD CONSTRAINT "FK_executions_signal"
      FOREIGN KEY ("signal_id") REFERENCES "signals"("id")
      ON DELETE CASCADE ON UPDATE NO ACTION
    `);

    await queryRunner.query(`
      ALTER TABLE "market_outcomes"
      ADD CONSTRAINT "FK_market_outcomes_market"
      FOREIGN KEY ("market_id") REFERENCES "markets"("id")
      ON DELETE CASCADE ON UPDATE NO ACTION
    `);

    // Create indexes for markets
    await queryRunner.query(`
      CREATE INDEX "IDX_markets_category_status_resolution"
      ON "markets" ("category", "status", "resolution_date")
    `);

    // Create indexes for signals
    await queryRunner.query(`
      CREATE INDEX "IDX_signals_market_status_urgent_created"
      ON "signals" ("market_id", "status", "is_urgent", "created_at")
    `);

    // Create indexes for executions
    await queryRunner.query(`
      CREATE INDEX "IDX_executions_signal_executed"
      ON "executions" ("signal_id", "executed_at")
    `);

    // Create indexes for market_outcomes
    await queryRunner.query(`
      CREATE INDEX "IDX_market_outcomes_market_resolved"
      ON "market_outcomes" ("market_id", "resolved_at")
    `);

    // Create unique constraint for external_id + source combination
    await queryRunner.query(`
      CREATE UNIQUE INDEX "IDX_markets_external_source"
      ON "markets" ("external_id", "source")
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Drop indexes
    await queryRunner.query(`DROP INDEX IF EXISTS "IDX_markets_external_source"`);
    await queryRunner.query(`DROP INDEX IF EXISTS "IDX_market_outcomes_market_resolved"`);
    await queryRunner.query(`DROP INDEX IF EXISTS "IDX_executions_signal_executed"`);
    await queryRunner.query(`DROP INDEX IF EXISTS "IDX_signals_market_status_urgent_created"`);
    await queryRunner.query(`DROP INDEX IF EXISTS "IDX_markets_category_status_resolution"`);

    // Drop foreign key constraints
    await queryRunner.query(`ALTER TABLE "market_outcomes" DROP CONSTRAINT IF EXISTS "FK_market_outcomes_market"`);
    await queryRunner.query(`ALTER TABLE "executions" DROP CONSTRAINT IF EXISTS "FK_executions_signal"`);
    await queryRunner.query(`ALTER TABLE "signals" DROP CONSTRAINT IF EXISTS "FK_signals_market"`);

    // Drop tables
    await queryRunner.query(`DROP TABLE IF EXISTS "market_outcomes"`);
    await queryRunner.query(`DROP TABLE IF EXISTS "executions"`);
    await queryRunner.query(`DROP TABLE IF EXISTS "signals"`);
    await queryRunner.query(`DROP TABLE IF EXISTS "markets"`);
  }
}
