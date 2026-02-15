import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  JoinColumn,
  Index,
} from 'typeorm';
import { Signal } from '../../signals/entities/signal.entity';

@Entity('executions')
@Index(['signalId', 'executedAt'])
export class Execution {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', name: 'signal_id' })
  signalId: string;

  @ManyToOne(() => Signal, (signal) => signal.executions, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'signal_id' })
  signal: Signal;

  @Column({ type: 'datetime', name: 'executed_at' })
  executedAt: Date;

  @Column({ type: 'decimal', precision: 10, scale: 4, name: 'entry_price_real' })
  entryPriceReal: number;

  @Column({ type: 'decimal', precision: 18, scale: 2, name: 'stake_amount' })
  stakeAmount: number;

  @Column({ type: 'text', nullable: true })
  notes: string | null;
}
