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
import { SignalsService } from './signals.service';
import { CreateSignalDto } from './dto/create-signal.dto';
import { UpdateSignalDto } from './dto/update-signal.dto';
import { QuerySignalDto } from './dto/query-signal.dto';
import { SignalResponseDto } from './dto/signal-response.dto';

@ApiTags('signals')
@Controller('signals')
export class SignalsController {
  constructor(private readonly signalsService: SignalsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new trading signal' })
  @ApiResponse({
    status: 201,
    description: 'Signal created successfully',
    type: SignalResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data or market not found' })
  create(@Body() createSignalDto: CreateSignalDto) {
    return this.signalsService.create(createSignalDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all signals with optional filters' })
  @ApiResponse({
    status: 200,
    description: 'Paginated list of signals with market info',
  })
  @ApiQuery({ name: 'category', required: false, description: 'Filter by market category' })
  @ApiQuery({ name: 'status', required: false, description: 'Filter by signal status' })
  @ApiQuery({ name: 'isUrgent', required: false, description: 'Filter urgent signals' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of results (default: 20)' })
  @ApiQuery({ name: 'offset', required: false, description: 'Offset for pagination (default: 0)' })
  findAll(@Query() query: QuerySignalDto) {
    return this.signalsService.findAll(query);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a single signal by ID' })
  @ApiParam({ name: 'id', description: 'Signal UUID' })
  @ApiResponse({
    status: 200,
    description: 'Signal found',
    type: SignalResponseDto,
  })
  @ApiResponse({ status: 404, description: 'Signal not found' })
  findOne(@Param('id', ParseUUIDPipe) id: string) {
    return this.signalsService.findOne(id);
  }

  @Patch(':id')
  @ApiOperation({ summary: 'Update a signal' })
  @ApiParam({ name: 'id', description: 'Signal UUID' })
  @ApiResponse({
    status: 200,
    description: 'Signal updated successfully',
    type: SignalResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 404, description: 'Signal not found' })
  update(
    @Param('id', ParseUUIDPipe) id: string,
    @Body() updateSignalDto: UpdateSignalDto,
  ) {
    return this.signalsService.update(id, updateSignalDto);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete a signal' })
  @ApiParam({ name: 'id', description: 'Signal UUID' })
  @ApiResponse({ status: 204, description: 'Signal deleted successfully' })
  @ApiResponse({ status: 404, description: 'Signal not found' })
  remove(@Param('id', ParseUUIDPipe) id: string) {
    return this.signalsService.remove(id);
  }
}
