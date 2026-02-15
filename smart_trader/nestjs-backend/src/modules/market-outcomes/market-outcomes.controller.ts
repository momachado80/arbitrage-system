import {
  Controller,
  Get,
  Post,
  Body,
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
import { MarketOutcomesService } from './market-outcomes.service';
import { CreateMarketOutcomeDto } from './dto/create-market-outcome.dto';
import { QueryMarketOutcomeDto } from './dto/query-market-outcome.dto';
import { MarketOutcomeResponseDto } from './dto/market-outcome-response.dto';

@ApiTags('market-outcomes')
@Controller('market-outcomes')
export class MarketOutcomesController {
  constructor(private readonly marketOutcomesService: MarketOutcomesService) {}

  @Post()
  @ApiOperation({ summary: 'Record a market resolution/outcome' })
  @ApiResponse({
    status: 201,
    description: 'Market outcome recorded successfully',
    type: MarketOutcomeResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data or market not found' })
  @ApiResponse({ status: 409, description: 'Market already has an outcome' })
  create(@Body() createDto: CreateMarketOutcomeDto) {
    return this.marketOutcomesService.create(createDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all market outcomes with optional filters' })
  @ApiResponse({
    status: 200,
    description: 'Paginated list of market outcomes',
  })
  @ApiQuery({ name: 'marketId', required: false, description: 'Filter by market ID' })
  @ApiQuery({ name: 'outcome', required: false, description: 'Filter by outcome (won/lost/push)' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of results (default: 20)' })
  @ApiQuery({ name: 'offset', required: false, description: 'Offset for pagination (default: 0)' })
  findAll(@Query() query: QueryMarketOutcomeDto) {
    return this.marketOutcomesService.findAll(query);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a single market outcome by ID' })
  @ApiParam({ name: 'id', description: 'Market outcome UUID' })
  @ApiResponse({
    status: 200,
    description: 'Market outcome found',
    type: MarketOutcomeResponseDto,
  })
  @ApiResponse({ status: 404, description: 'Market outcome not found' })
  findOne(@Param('id', ParseUUIDPipe) id: string) {
    return this.marketOutcomesService.findOne(id);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete a market outcome' })
  @ApiParam({ name: 'id', description: 'Market outcome UUID' })
  @ApiResponse({ status: 204, description: 'Market outcome deleted successfully' })
  @ApiResponse({ status: 404, description: 'Market outcome not found' })
  remove(@Param('id', ParseUUIDPipe) id: string) {
    return this.marketOutcomesService.remove(id);
  }
}
