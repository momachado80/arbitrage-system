import {
  Injectable,
  NotFoundException,
  BadRequestException,
  ConflictException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { MarketOutcome } from './entities/market-outcome.entity';
import { CreateMarketOutcomeDto } from './dto/create-market-outcome.dto';
import { QueryMarketOutcomeDto } from './dto/query-market-outcome.dto';
import { PaginatedResponseDto } from '../../common/dto/pagination.dto';
import { MarketsService } from '../markets/markets.service';
import { MarketStatus } from '../../common/enums';

@Injectable()
export class MarketOutcomesService {
  constructor(
    @InjectRepository(MarketOutcome)
    private readonly marketOutcomeRepository: Repository<MarketOutcome>,
    private readonly marketsService: MarketsService,
  ) {}

  /**
   * Create a new market outcome (record resolution)
   */
  async create(createDto: CreateMarketOutcomeDto): Promise<MarketOutcome> {
    // Verify market exists
    const market = await this.marketsService.findOne(createDto.marketId);
    if (!market) {
      throw new BadRequestException(
        `Market with ID '${createDto.marketId}' not found`,
      );
    }

    // Check if market already has an outcome
    const existingOutcome = await this.marketOutcomeRepository.findOne({
      where: { marketId: createDto.marketId },
    });

    if (existingOutcome) {
      throw new ConflictException(
        `Market '${createDto.marketId}' already has an outcome recorded`,
      );
    }

    // Create the outcome
    const outcome = this.marketOutcomeRepository.create({
      ...createDto,
      resolvedAt: new Date(createDto.resolvedAt),
    });

    const saved = await this.marketOutcomeRepository.save(outcome);

    // Update market status to resolved
    await this.marketsService.update(createDto.marketId, {
      status: MarketStatus.RESOLVED,
    });

    return this.findOne(saved.id);
  }

  /**
   * Find all market outcomes with optional filters and pagination
   */
  async findAll(
    query: QueryMarketOutcomeDto,
  ): Promise<PaginatedResponseDto<MarketOutcome>> {
    const { limit = 20, offset = 0, marketId, outcome } = query;

    const queryBuilder = this.marketOutcomeRepository
      .createQueryBuilder('marketOutcome')
      .leftJoinAndSelect('marketOutcome.market', 'market');

    // Filter by market ID
    if (marketId) {
      queryBuilder.andWhere('marketOutcome.marketId = :marketId', { marketId });
    }

    // Filter by outcome
    if (outcome) {
      queryBuilder.andWhere('marketOutcome.outcome = :outcome', { outcome });
    }

    queryBuilder
      .orderBy('marketOutcome.resolvedAt', 'DESC')
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
   * Find a single market outcome by ID
   */
  async findOne(id: string): Promise<MarketOutcome> {
    const outcome = await this.marketOutcomeRepository.findOne({
      where: { id },
      relations: ['market'],
    });

    if (!outcome) {
      throw new NotFoundException(`Market outcome with ID '${id}' not found`);
    }

    return outcome;
  }

  /**
   * Find outcome by market ID
   */
  async findByMarketId(marketId: string): Promise<MarketOutcome | null> {
    return this.marketOutcomeRepository.findOne({
      where: { marketId },
      relations: ['market'],
    });
  }

  /**
   * Delete a market outcome
   */
  async remove(id: string): Promise<void> {
    const outcome = await this.findOne(id);

    // Optionally revert market status back to open
    await this.marketsService.update(outcome.marketId, {
      status: MarketStatus.OPEN,
    });

    await this.marketOutcomeRepository.remove(outcome);
  }

  /**
   * Get win rate by category
   */
  async getWinRateByCategory(): Promise<
    { category: string; wins: number; total: number; winRate: number }[]
  > {
    const results = await this.marketOutcomeRepository
      .createQueryBuilder('outcome')
      .leftJoin('outcome.market', 'market')
      .select('market.category', 'category')
      .addSelect(
        "SUM(CASE WHEN outcome.outcome = 'won' THEN 1 ELSE 0 END)",
        'wins',
      )
      .addSelect('COUNT(*)', 'total')
      .groupBy('market.category')
      .getRawMany();

    return results.map((r) => ({
      category: r.category,
      wins: parseInt(r.wins, 10),
      total: parseInt(r.total, 10),
      winRate:
        r.total > 0 ? Math.round((parseInt(r.wins, 10) / parseInt(r.total, 10)) * 100) : 0,
    }));
  }
}
