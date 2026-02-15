import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  ManyToOne,
  OneToMany,
  JoinColumn,
  Index,
} from 'typeorm';
import { SignalStatus, TradeDirection } from '../../../common/enums';
import { Market } from '../../markets/entities/market.entity';
import { Execution } from '../../executions/entities/execution.entity';

@Entity('signals')
@Index(['marketId', 'status', 'isUrgent', 'createdAt'])
export class Signal {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', name: 'market_id' })
  marketId: string;

  @ManyToOne(() => Market, (market) => market.signals, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'market_id' })
  market: Market;

  @CreateDateColumn({ name: 'created_at' })
  createdAt: Date;

  @Column({ type: 'varchar', length: 20 })
  direction: TradeDirection;

  @Column({ type: 'integer', name: 'confidence_score' })
  confidenceScore: number;

  @Column({ type: 'decimal', precision: 10, scale: 4, name: 'entry_price_suggested' })
  entryPriceSuggested: number;

  @Column({ type: 'decimal', precision: 10, scale: 4, name: 'target_price' })
  targetPrice: number;

  @Column({ type: 'decimal', precision: 10, scale: 4, name: 'stop_price' })
  stopPrice: number;

  @Column({ type: 'text', nullable: true })
  rationale: string | null;

  @Column({ type: 'boolean', name: 'is_urgent', default: false })
  isUrgent: boolean;

  @Column({ type: 'varchar', length: 20, default: SignalStatus.ACTIVE })
  status: SignalStatus;

  // Relations
  @OneToMany(() => Execution, (execution) => execution.signal)
  executions: Execution[];
}
