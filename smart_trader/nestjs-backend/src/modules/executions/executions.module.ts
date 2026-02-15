import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ExecutionsService } from './executions.service';
import { ExecutionsController } from './executions.controller';
import { Execution } from './entities/execution.entity';
import { SignalsModule } from '../signals/signals.module';

@Module({
  imports: [TypeOrmModule.forFeature([Execution]), SignalsModule],
  controllers: [ExecutionsController],
  providers: [ExecutionsService],
  exports: [ExecutionsService],
})
export class ExecutionsModule {}
