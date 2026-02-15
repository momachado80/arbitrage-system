import {
  Injectable,
  NotFoundException,
  ConflictException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, FindOptionsWhere, LessThanOrEqual, MoreThanOrEqual } from 'typeorm';
import { Market } from './entities/market.entity';
import { CreateMarketDto } from './dto/create-market.dto';
import { UpdateMarketDto } from './dto/update-market.dto';
import { QueryMarketDto } from './dto/query-market.dto';
import { PaginatedResponseDto } from '../../common/dto/pagination.dto';

@Injectable()
export class MarketsService {
  constructor(
    @InjectRepository(Market)
    private readonly marketRepository: Repository<Market>,
  ) {}

  /**
   * Create a new market
   */
  async create(createMarketDto: CreateMarketDto): Promise<Market> {
    // Check if market with same externalId and source already exists
    const existing = await this.marketRepository.findOne({
      where: {
        externalId: createMarketDto.externalId,
        source: createMarketDto.source,
      },
    });

    if (existing) {
      throw new ConflictException(
        `Market with externalId '${createMarketDto.externalId}' from source '${createMarketDto.source}' already exists`,
      );
    }

    const market = this.marketRepository.create({
      ...createMarketDto,
      resolutionDate: createMarketDto.resolutionDate
        ? new Date(createMarketDto.resolutionDate)
        : null,
    });

    return this.marketRepository.save(market);
  }

  /**
   * Find all markets with optional filters and pagination
   */
  async findAll(query: QueryMarketDto): Promise<PaginatedResponseDto<Market>> {
    const { limit = 20, offset = 0, category, status, resolutionBefore, resolutionAfter } = query;

    const where: FindOptionsWhere<Market> = {};

    if (category) {
      where.category = category;
    }

    if (status) {
      where.status = status;
    }

    // Build query with date filters
    const queryBuilder = this.marketRepository.createQueryBuilder('market');
    
    if (category) {
      queryBuilder.andWhere('market.category = :category', { category });
    }
    
    if (status) {
      queryBuilder.andWhere('market.status = :status', { status });
    }

    if (resolutionBefore) {
      queryBuilder.andWhere('market.resolutionDate <= :resolutionBefore', {
        resolutionBefore: new Date(resolutionBefore),
      });
    }

    if (resolutionAfter) {
      queryBuilder.andWhere('market.resolutionDate >= :resolutionAfter', {
        resolutionAfter: new Date(resolutionAfter),
      });
    }

    queryBuilder
      .orderBy('market.createdAt', 'DESC')
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
   * Find a single market by ID
   */
  async findOne(id: string): Promise<Market> {
    const market = await this.marketRepository.findOne({
      where: { id },
      relations: ['signals', 'outcomes'],
    });

    if (!market) {
      throw new NotFoundException(`Market with ID '${id}' not found`);
    }

    return market;
  }

  /**
   * Update a market
   */
  async update(id: string, updateMarketDto: UpdateMarketDto): Promise<Market> {
    const market = await this.findOne(id);

    const { resolutionDate, ...rest } = updateMarketDto;
    const updateData: Partial<Market> = {
      ...rest,
    };

    if (resolutionDate !== undefined) {
      updateData.resolutionDate = resolutionDate ? new Date(resolutionDate) : null;
    }

    Object.assign(market, updateData);
    return this.marketRepository.save(market);
  }

  /**
   * Delete a market (hard delete)
   * 
   * Note: We use hard delete here for simplicity in Phase 1.
   * In production, you might want to implement soft delete by adding
   * a `deletedAt` column and using TypeORM's @DeleteDateColumn().
   * This would preserve data integrity for audit purposes.
   */
  async remove(id: string): Promise<void> {
    const market = await this.findOne(id);
    await this.marketRepository.remove(market);
  }

  /**
   * Find market by external ID and source
   */
  async findByExternalId(externalId: string, source: string): Promise<Market | null> {
    return this.marketRepository.findOne({
      where: { externalId, source },
    });
  }
}
