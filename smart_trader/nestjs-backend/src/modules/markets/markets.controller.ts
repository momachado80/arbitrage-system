import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  HttpCode,
  HttpStatus,
  ParseUUIDPipe,
} from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiResponse,
  ApiParam,
  ApiQuery,
} from '@nestjs/swagger';
import { MarketsService } from './markets.service';
import { CreateMarketDto } from './dto/create-market.dto';
import { UpdateMarketDto } from './dto/update-market.dto';
import { QueryMarketDto } from './dto/query-market.dto';
import { MarketResponseDto } from './dto/market-response.dto';

@ApiTags('markets')
@Controller('markets')
export class MarketsController {
  constructor(private readonly marketsService: MarketsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new market' })
  @ApiResponse({
    status: 201,
    description: 'Market created successfully',
    type: MarketResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 409, description: 'Market already exists' })
  create(@Body() createMarketDto: CreateMarketDto) {
    return this.marketsService.create(createMarketDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all markets with optional filters' })
  @ApiResponse({
    status: 200,
    description: 'Paginated list of markets',
  })
  @ApiQuery({ name: 'category', required: false, description: 'Filter by category' })
  @ApiQuery({ name: 'status', required: false, description: 'Filter by status' })
  @ApiQuery({ name: 'resolutionBefore', required: false, description: 'Filter by resolution date (before)' })
  @ApiQuery({ name: 'resolutionAfter', required: false, description: 'Filter by resolution date (after)' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of results (default: 20)' })
  @ApiQuery({ name: 'offset', required: false, description: 'Offset for pagination (default: 0)' })
  findAll(@Query() query: QueryMarketDto) {
    return this.marketsService.findAll(query);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a single market by ID' })
  @ApiParam({ name: 'id', description: 'Market UUID' })
  @ApiResponse({
    status: 200,
    description: 'Market found',
    type: MarketResponseDto,
  })
  @ApiResponse({ status: 404, description: 'Market not found' })
  findOne(@Param('id', ParseUUIDPipe) id: string) {
    return this.marketsService.findOne(id);
  }

  @Patch(':id')
  @ApiOperation({ summary: 'Update a market' })
  @ApiParam({ name: 'id', description: 'Market UUID' })
  @ApiResponse({
    status: 200,
    description: 'Market updated successfully',
    type: MarketResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 404, description: 'Market not found' })
  update(
    @Param('id', ParseUUIDPipe) id: string,
    @Body() updateMarketDto: UpdateMarketDto,
  ) {
    return this.marketsService.update(id, updateMarketDto);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete a market' })
  @ApiParam({ name: 'id', description: 'Market UUID' })
  @ApiResponse({ status: 204, description: 'Market deleted successfully' })
  @ApiResponse({ status: 404, description: 'Market not found' })
  remove(@Param('id', ParseUUIDPipe) id: string) {
    return this.marketsService.remove(id);
  }
}
