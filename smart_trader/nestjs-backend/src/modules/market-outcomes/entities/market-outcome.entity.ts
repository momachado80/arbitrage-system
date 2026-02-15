import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  JoinColumn,
  Index,
} from 'typeorm';
import { MarketOutcomeResult } from '../../../common/enums';
import { Market } from '../../markets/entities/market.entity';

@Entity('market_outcomes')
@Index(['marketId', 'resolvedAt'])
export class MarketOutcome {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', name: 'market_id' })
  marketId: string;

  @ManyToOne(() => Market, (market) => market.outcomes, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'market_id' })
  market: Market;

  @Column({ type: 'datetime', name: 'resolved_at' })
  resolvedAt: Date;

  @Column({ type: 'varchar', length: 20 })
  outcome: MarketOutcomeResult;

  @Column({ type: 'decimal', precision: 10, scale: 4, name: 'final_price' })
  finalPrice: number;

  @Column({ type: 'text', nullable: true })
  notes: string | null;
}
