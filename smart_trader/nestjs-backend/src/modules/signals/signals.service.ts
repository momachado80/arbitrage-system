import {
  Injectable,
  NotFoundException,
  BadRequestException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Signal } from './entities/signal.entity';
import { CreateSignalDto } from './dto/create-signal.dto';
import { UpdateSignalDto } from './dto/update-signal.dto';
import { QuerySignalDto } from './dto/query-signal.dto';
import { PaginatedResponseDto } from '../../common/dto/pagination.dto';
import { MarketsService } from '../markets/markets.service';

@Injectable()
export class SignalsService {
  constructor(
    @InjectRepository(Signal)
    private readonly signalRepository: Repository<Signal>,
    private readonly marketsService: MarketsService,
  ) {}

  /**
   * Create a new signal
   */
  async create(createSignalDto: CreateSignalDto): Promise<Signal> {
    // Verify market exists
    const market = await this.marketsService.findOne(createSignalDto.marketId);
    if (!market) {
      throw new BadRequestException(
        `Market with ID '${createSignalDto.marketId}' not found`,
      );
    }

    const signal = this.signalRepository.create(createSignalDto);
    const saved = await this.signalRepository.save(signal);

    // Return with market relation loaded
    return this.findOne(saved.id);
  }

  /**
   * Find all signals with optional filters and pagination
   */
  async findAll(query: QuerySignalDto): Promise<PaginatedResponseDto<Signal>> {
    const { limit = 20, offset = 0, category, status, isUrgent } = query;

    const queryBuilder = this.signalRepository
      .createQueryBuilder('signal')
      .leftJoinAndSelect('signal.market', 'market');

    // Filter by market category
    if (category) {
      queryBuilder.andWhere('market.category = :category', { category });
    }

    // Filter by signal status
    if (status) {
      queryBuilder.andWhere('signal.status = :status', { status });
    }

    // Filter by urgent flag
    if (isUrgent !== undefined) {
      queryBuilder.andWhere('signal.isUrgent = :isUrgent', { isUrgent });
    }

    queryBuilder
      .orderBy('signal.createdAt', 'DESC')
      .skip(offset)
      .take(limit);

    const [data, total] = await queryBuilder.getManyAndCount();

    return {
      data,
      total,
      limit,
      offset,
      hasMore: offset + data.length < total,
    };
  }

  /**
   * Find a single signal by ID
   */
  async findOne(id: string): Promise<Signal> {
    const signal = await this.signalRepository.findOne({
      where: { id },
      relations: ['market', 'executions'],
    });

    if (!signal) {
      throw new NotFoundException(`Signal with ID '${id}' not found`);
    }

    return signal;
  }

  /**
   * Update a signal
   */
  async update(id: string, updateSignalDto: UpdateSignalDto): Promise<Signal> {
    const signal = await this.findOne(id);
    Object.assign(signal, updateSignalDto);
    await this.signalRepository.save(signal);
    return this.findOne(id);
  }

  /**
   * Delete a signal
   */
  async remove(id: string): Promise<void> {
    const signal = await this.findOne(id);
    await this.signalRepository.remove(signal);
  }

  /**
   * Find signals by market ID
   */
  async findByMarketId(marketId: string): Promise<Signal[]> {
    return this.signalRepository.find({
      where: { marketId },
      relations: ['market'],
      order: { createdAt: 'DESC' },
    });
  }

  /**
   * Count active signals
   */
  async countActive(): Promise<number> {
    return this.signalRepository.count({
      where: { status: 'active' as any },
    });
  }

  /**
   * Count urgent signals
   */
  async countUrgent(): Promise<number> {
    return this.signalRepository.count({
      where: { isUrgent: true, status: 'active' as any },
    });
  }
}
