import { Controller, Post, Get } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { DevService } from './dev.service';

/**
 * ⚠️  DEV CONTROLLER - DO NOT ENABLE IN PRODUCTION ⚠️
 * 
 * This controller provides development/testing utilities.
 * - POST /dev/seed: Populates database with demo data
 * - GET /dev/counts: Returns current row counts
 * 
 * These endpoints should be disabled or removed in production deployments.
 */
@ApiTags('dev')
@Controller('dev')
export class DevController {
  constructor(private readonly devService: DevService) {}

  @Post('seed')
  @ApiOperation({
    summary: '⚠️ DEV ONLY: Seed database with demo data',
    description: `
      **WARNING: This endpoint DELETES ALL EXISTING DATA before inserting demo data!**
      
      Creates:
      - 5 markets (politics, crypto, tech, sports, finance)
      - 7 signals (mix of YES/NO, various confidence levels)
      - 7 executions (trades with different stake amounts)
      - 4 market outcomes (2 won, 1 lost, 1 push)
      
      One market (Fed rate cut) is left OPEN without an outcome to demonstrate
      how the system handles unresolved trades in stats.
      
      **DO NOT USE IN PRODUCTION!**
    `,
  })
  @ApiResponse({
    status: 201,
    description: 'Database seeded successfully',
    schema: {
      type: 'object',
      properties: {
        message: { type: 'string', example: 'Dev seed completed' },
        created: {
          type: 'object',
          properties: {
            markets: { type: 'number', example: 5 },
            signals: { type: 'number', example: 7 },
            executions: { type: 'number', example: 7 },
            outcomes: { type: 'number', example: 4 },
          },
        },
      },
    },
  })
  async seed(): Promise<{ message: string; created: Record<string, number> }> {
    const created = await this.devService.seed();
    return {
      message: 'Dev seed completed',
      created,
    };
  }

  @Get('counts')
  @ApiOperation({
    summary: 'Get current row counts for all tables',
    description: 'Returns the number of rows in each core table. Useful for verification.',
  })
  @ApiResponse({
    status: 200,
    description: 'Current row counts',
    schema: {
      type: 'object',
      properties: {
        markets: { type: 'number', example: 5 },
        signals: { type: 'number', example: 7 },
        executions: { type: 'number', example: 7 },
        outcomes: { type: 'number', example: 4 },
      },
    },
  })
  async getCounts(): Promise<Record<string, number>> {
    return this.devService.getCounts();
  }
}
