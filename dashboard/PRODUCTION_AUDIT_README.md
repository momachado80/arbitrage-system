# Production Shadow Audit — URL Discovery

## 1. Qual serviço hospeda `/api/shadow/audit`?

O **Next.js dashboard** (`dashboard/`) com:
- `instrumentation.ts` → inicia `executionWorker`
- `executionWorker` → executa scanner + dispatcher + shadow simulation
- Rotas em `app/api/shadow/*`

Ou seja: o mesmo processo Next.js que serve a UI também roda a shadow simulation e expõe `/api/shadow/audit`.

## 2. URLs de produção conhecidas (do repo)

| URL | Serviço | Evidência |
|-----|---------|-----------|
| `https://web-production-800bb.up.railway.app` | Python (uvicorn) | Procfile, `/analytics` retorna JSON Python |
| `https://web-production-eca1e.up.railway.app` | Python (frontend) | `frontend/*.html` usam como API_URL |

Ambas retornam 404 em `/api/shadow/audit` — são Python, não Next.js.

O dashboard Next.js é um serviço separado e sua URL **não** está no repositório.

## 3. Como expor a URL sem busca manual

### Opção A: Arquivo `.production-url`
```bash
echo "https://SEU-DASHBOARD-NEXTJS.up.railway.app" > dashboard/.production-url
```
Depois:
```bash
cd dashboard && npx ts-node -P tsconfig.worker.json scripts/get-production-audit.ts
```

### Opção B: Variável de ambiente
```bash
AUDIT_URL=https://SEU-DASHBOARD.up.railway.app npx ts-node -P tsconfig.worker.json dashboard/scripts/get-production-audit.ts
```

### Opção C: Railway CLI (quando conectado)
```bash
cd dashboard && railway link  # vincule ao projeto
railway status               # mostra serviço atual
railway domain               # mostra domínio do serviço
```
Em seguida use a URL retornada em `AUDIT_URL` ou `.production-url`.

### Opção D: Log de boot no Railway
O Next.js usa `RAILWAY_PUBLIC_DOMAIN` em produção. Em `/api/deployment-info`:
```json
{ "dashboardUrl": "https://xxx.up.railway.app" }
```
Depois de descobrir a URL (por ex. no painel do Railway), chame esse endpoint para confirmar.

## 4. Script que tenta descobrir a URL

`scripts/get-production-audit.ts`:
1. Usa `AUDIT_URL` ou `DASHBOARD_URL`
2. Lê `dashboard/.production-url` se existir
3. Testa candidatos 800bb e eca1e (nenhum é Next.js)

Adicione `.production-url` com a URL do seu dashboard Next.js no Railway para o script funcionar.
