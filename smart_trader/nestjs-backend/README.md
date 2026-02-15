# Smart Event Trader API - Phase 1

Backend API para o Smart Event Trader usando NestJS + TypeORM + PostgreSQL.

## Tech Stack

- **Framework**: NestJS 10.x
- **ORM**: TypeORM 0.3.x
- **Database**: PostgreSQL
- **Validation**: class-validator + class-transformer
- **Documentation**: Swagger/OpenAPI

## Setup

### 1. Instalar dependências

```bash
cd smart_trader/nestjs-backend
npm install
```

### 2. Configurar banco de dados

Crie um arquivo `.env` baseado no `.env.example`:

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais do PostgreSQL:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=sua_senha_aqui
DB_DATABASE=smart_event_trader

PORT=3000
NODE_ENV=development
```

### 3. Criar o banco de dados

```bash
# Conecte ao PostgreSQL e crie o banco
psql -U postgres -c "CREATE DATABASE smart_event_trader;"
```

### 4. Executar migrations

```bash
npm run migration:run
```

### 5. Iniciar o servidor

```bash
# Modo desenvolvimento (com hot reload)
npm run start:dev

# Modo produção
npm run build
npm run start:prod
```

O servidor estará disponível em:
- **API**: http://localhost:3000
- **Swagger Docs**: http://localhost:3000/api

## Estrutura de Diretórios

```
src/
├── common/
│   ├── dto/           # DTOs compartilhados (pagination)
│   ├── enums/         # Enums (MarketStatus, SignalStatus, etc.)
│   └── filters/       # Exception filters
├── database/
│   ├── data-source.ts # Configuração TypeORM
│   └── migrations/    # Migration files
├── modules/
│   ├── markets/       # Módulo de mercados
│   ├── signals/       # Módulo de sinais
│   ├── executions/    # Módulo de execuções
│   └── market-outcomes/ # Módulo de resultados
├── app.module.ts      # Módulo principal
└── main.ts            # Entry point
```

## API Endpoints

### Markets

| Method | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/markets` | Criar mercado |
| GET | `/markets` | Listar mercados (com filtros) |
| GET | `/markets/:id` | Buscar mercado por ID |
| PATCH | `/markets/:id` | Atualizar mercado |
| DELETE | `/markets/:id` | Deletar mercado |

### Signals

| Method | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/signals` | Criar sinal |
| GET | `/signals` | Listar sinais (com filtros) |
| GET | `/signals/:id` | Buscar sinal por ID |
| PATCH | `/signals/:id` | Atualizar sinal |
| DELETE | `/signals/:id` | Deletar sinal |

### Executions

| Method | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/executions` | Registrar execução |
| GET | `/executions` | Listar execuções (com filtros) |
| GET | `/executions/:id` | Buscar execução por ID |
| DELETE | `/executions/:id` | Deletar execução |

### Market Outcomes

| Method | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/market-outcomes` | Registrar resultado |
| GET | `/market-outcomes` | Listar resultados |
| GET | `/market-outcomes/:id` | Buscar resultado por ID |
| DELETE | `/market-outcomes/:id` | Deletar resultado |

## Exemplos de Uso (curl)

### 1. Criar um Market

```bash
curl -X POST http://localhost:3000/markets \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Trump wins 2028 election",
    "source": "polymarket",
    "externalId": "0x1234567890abcdef",
    "category": "politics",
    "resolutionDate": "2028-11-05T00:00:00Z",
    "status": "open"
  }'
```

Resposta:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Trump wins 2028 election",
  "source": "polymarket",
  "externalId": "0x1234567890abcdef",
  "category": "politics",
  "resolutionDate": "2028-11-05T00:00:00.000Z",
  "status": "open",
  "createdAt": "2024-01-30T10:00:00.000Z",
  "updatedAt": "2024-01-30T10:00:00.000Z"
}
```

### 2. Criar um Signal para o Market

```bash
curl -X POST http://localhost:3000/signals \
  -H "Content-Type: application/json" \
  -d '{
    "marketId": "550e8400-e29b-41d4-a716-446655440000",
    "direction": "YES",
    "confidenceScore": 75,
    "entryPriceSuggested": 0.45,
    "targetPrice": 0.65,
    "stopPrice": 0.35,
    "rationale": "Strong momentum detected + whale accumulation",
    "isUrgent": true
  }'
```

### 3. Registrar uma Execution

```bash
curl -X POST http://localhost:3000/executions \
  -H "Content-Type: application/json" \
  -d '{
    "signalId": "SIGNAL_UUID_HERE",
    "executedAt": "2024-01-30T10:15:00Z",
    "entryPriceReal": 0.47,
    "stakeAmount": 100.00,
    "notes": "Executed via Polymarket web interface"
  }'
```

### 4. Registrar um Market Outcome

```bash
curl -X POST http://localhost:3000/market-outcomes \
  -H "Content-Type: application/json" \
  -d '{
    "marketId": "550e8400-e29b-41d4-a716-446655440000",
    "resolvedAt": "2028-11-06T00:00:00Z",
    "outcome": "won",
    "finalPrice": 1.0,
    "notes": "Market resolved YES after election results confirmed"
  }'
```

### 5. Listar Signals com Filtros

```bash
# Todos os sinais urgentes de política
curl "http://localhost:3000/signals?category=politics&isUrgent=true&limit=10"

# Sinais ativos
curl "http://localhost:3000/signals?status=active"
```

### 6. Listar Markets com Filtros

```bash
# Mercados de política que resolvem até fim de 2028
curl "http://localhost:3000/markets?category=politics&resolutionBefore=2028-12-31T23:59:59Z"
```

## Migrations

```bash
# Executar migrations pendentes
npm run migration:run

# Reverter última migration
npm run migration:revert

# Gerar nova migration (após mudar entities)
npm run migration:generate src/database/migrations/NomeDaMigration
```

## Validação

Todas as rotas usam `class-validator` para validação:

- **400 Bad Request**: Payload inválido
- **404 Not Found**: Recurso não encontrado
- **409 Conflict**: Recurso já existe (ex: market com mesmo externalId)

## Próximas Fases

- **Phase 2**: Stats e agregações (win rate, ROI, etc.)
- **Phase 3**: Risk management endpoints
- **Phase 4**: Backtesting engine
- **Phase 5**: Real-time WebSocket updates
