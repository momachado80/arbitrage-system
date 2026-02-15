import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  OneToMany,
  Index,
} from 'typeorm';
import { MarketStatus, MarketCategory } from '../../../common/enums';
import { Signal } from '../../signals/entities/signal.entity';
import { MarketOutcome } from '../../market-outcomes/entities/market-outcome.entity';

@Entity('markets')
@Index(['category', 'status', 'resolutionDate'])
export class Market {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 500 })
  title: string;

  @Column({ type: 'varchar', length: 100 })
  source: string;

  @Column({ type: 'varchar', length: 255, name: 'external_id' })
  externalId: string;

  @Column({
    type: 'varchar',
    length: 50,
    default: MarketCategory.OTHER,
  })
  category: string;

  @Column({
    type: 'datetime',
    name: 'resolution_date',
    nullable: true,
  })
  resolutionDate: Date | null;

  @Column({
    type: 'varchar',
    length: 20,
    default: MarketStatus.OPEN,
  })
  status: MarketStatus;

  @CreateDateColumn({ name: 'created_at' })
  createdAt: Date;

  @UpdateDateColumn({ name: 'updated_at' })
  updatedAt: Date;

  // Relations
  @OneToMany(() => Signal, (signal) => signal.market)
  signals: Signal[];

  @OneToMany(() => MarketOutcome, (outcome) => outcome.market)
  outcomes: MarketOutcome[];
}
