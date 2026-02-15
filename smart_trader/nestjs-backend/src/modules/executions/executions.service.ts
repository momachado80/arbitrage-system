import {
  Injectable,
  NotFoundException,
  BadRequestException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Execution } from './entities/execution.entity';
import { CreateExecutionDto } from './dto/create-execution.dto';
import { QueryExecutionDto } from './dto/query-execution.dto';
import { PaginatedResponseDto } from '../../common/dto/pagination.dto';
import { SignalsService } from '../signals/signals.service';

@Injectable()
export class ExecutionsService {
  constructor(
    @InjectRepository(Execution)
    private readonly executionRepository: Repository<Execution>,
    private readonly signalsService: SignalsService,
  ) {}

  /**
   * Create a new execution (record a trade)
   */
  async create(createExecutionDto: CreateExecutionDto): Promise<Execution> {
    // Verify signal exists
    const signal = await this.signalsService.findOne(createExecutionDto.signalId);
    if (!signal) {
      throw new BadRequestException(
        `Signal with ID '${createExecutionDto.signalId}' not found`,
      );
    }

    const execution = this.executionRepository.create({
      ...createExecutionDto,
      executedAt: new Date(createExecutionDto.executedAt),
    });

    const saved = await this.executionRepository.save(execution);
    return this.findOne(saved.id);
  }

  /**
   * Find all executions with optional filters and pagination
   */
  async findAll(
    query: QueryExecutionDto,
  ): Promise<PaginatedResponseDto<Execution>> {
    const { limit = 20, offset = 0, signalId, fromDate, toDate } = query;

    const queryBuilder = this.executionRepository
      .createQueryBuilder('execution')
      .leftJoinAndSelect('execution.signal', 'signal')
      .leftJoinAndSelect('signal.market', 'market');

    // Filter by signal ID
    if (signalId) {
      queryBuilder.andWhere('execution.signalId = :signalId', { signalId });
    }

    // Filter by date range
    if (fromDate) {
      queryBuilder.andWhere('execution.executedAt >= :fromDate', {
        fromDate: new Date(fromDate),
      });
    }

    if (toDate) {
      queryBuilder.andWhere('execution.executedAt <= :toDate', {
        toDate: new Date(toDate),
      });
    }

    queryBuilder
      .orderBy('execution.executedAt', 'DESC')
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
   * Find a single execution by ID
   */
  async findOne(id: string): Promise<Execution> {
    const execution = await this.executionRepository.findOne({
      where: { id },
      relations: ['signal', 'signal.market'],
    });

    if (!execution) {
      throw new NotFoundException(`Execution with ID '${id}' not found`);
    }

    return execution;
  }

  /**
   * Delete an execution
   */
  async remove(id: string): Promise<void> {
    const execution = await this.findOne(id);
    await this.executionRepository.remove(execution);
  }

  /**
   * Find executions by signal ID
   */
  async findBySignalId(signalId: string): Promise<Execution[]> {
    return this.executionRepository.find({
      where: { signalId },
      relations: ['signal', 'signal.market'],
      order: { executedAt: 'DESC' },
    });
  }

  /**
   * Get total stake amount for a date range
   */
  async getTotalStake(fromDate?: Date, toDate?: Date): Promise<number> {
    const queryBuilder = this.executionRepository
      .createQueryBuilder('execution')
      .select('SUM(execution.stakeAmount)', 'total');

    if (fromDate) {
      queryBuilder.andWhere('execution.executedAt >= :fromDate', { fromDate });
    }

    if (toDate) {
      queryBuilder.andWhere('execution.executedAt <= :toDate', { toDate });
    }

    const result = await queryBuilder.getRawOne();
    return parseFloat(result?.total || '0');
  }
}
