import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Global validation pipe with class-validator
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: {
        enableImplicitConversion: true,
      },
    }),
  );

  // Enable CORS for frontend
  app.enableCors({
    origin: '*',
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS',
    credentials: true,
  });

  // Swagger/OpenAPI setup
  const config = new DocumentBuilder()
    .setTitle('Smart Event Trader API')
    .setDescription(
      'Backend API for Smart Event Trader - Real-time trading signals for prediction markets',
    )
    .setVersion('1.0')
    .addTag('markets', 'Market management endpoints')
    .addTag('signals', 'Trading signal endpoints')
    .addTag('executions', 'Trade execution endpoints')
    .addTag('market-outcomes', 'Market outcome/resolution endpoints')
    .addTag('stats', 'Trading statistics and performance metrics')
    .addTag('dev', '⚠️ Development utilities - DO NOT USE IN PRODUCTION')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api', app, document);

  const port = process.env.PORT || 3000;
  await app.listen(port);

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║           SMART EVENT TRADER API - Phase 2                   ║
╠══════════════════════════════════════════════════════════════╣
║  Server running on: http://localhost:${port}                    ║
║  Swagger docs:      http://localhost:${port}/api                ║
╚══════════════════════════════════════════════════════════════╝
  `);
}

bootstrap();
