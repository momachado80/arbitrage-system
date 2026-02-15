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
import { ExecutionsService } from './executions.service';
import { CreateExecutionDto } from './dto/create-execution.dto';
import { QueryExecutionDto } from './dto/query-execution.dto';
import { ExecutionResponseDto } from './dto/execution-response.dto';

@ApiTags('executions')
@Controller('executions')
export class ExecutionsController {
  constructor(private readonly executionsService: ExecutionsService) {}

  @Post()
  @ApiOperation({ summary: 'Record a trade execution' })
  @ApiResponse({
    status: 201,
    description: 'Execution recorded successfully',
    type: ExecutionResponseDto,
  })
  @ApiResponse({ status: 400, description: 'Invalid input data or signal not found' })
  create(@Body() createExecutionDto: CreateExecutionDto) {
    return this.executionsService.create(createExecutionDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all executions with optional filters' })
  @ApiResponse({
    status: 200,
    description: 'Paginated list of executions with signal and market info',
  })
  @ApiQuery({ name: 'signalId', required: false, description: 'Filter by signal ID' })
  @ApiQuery({ name: 'fromDate', required: false, description: 'Filter from date (ISO 8601)' })
  @ApiQuery({ name: 'toDate', required: false, description: 'Filter to date (ISO 8601)' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of results (default: 20)' })
  @ApiQuery({ name: 'offset', required: false, description: 'Offset for pagination (default: 0)' })
  findAll(@Query() query: QueryExecutionDto) {
    return this.executionsService.findAll(query);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a single execution by ID' })
  @ApiParam({ name: 'id', description: 'Execution UUID' })
  @ApiResponse({
    status: 200,
    description: 'Execution found',
    type: ExecutionResponseDto,
  })
  @ApiResponse({ status: 404, description: 'Execution not found' })
  findOne(@Param('id', ParseUUIDPipe) id: string) {
    return this.executionsService.findOne(id);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete an execution' })
  @ApiParam({ name: 'id', description: 'Execution UUID' })
  @ApiResponse({ status: 204, description: 'Execution deleted successfully' })
  @ApiResponse({ status: 404, description: 'Execution not found' })
  remove(@Param('id', ParseUUIDPipe) id: string) {
    return this.executionsService.remove(id);
  }
}
