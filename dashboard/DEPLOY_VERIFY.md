# Dashboard Deploy Verification

## Railway UI (OBRIGATÓRIO — conferir)

- **Service:** dashboard-next
- **Source > Root Directory:** `dashboard` (se vazio ou `.`, o build usa o repo root e falha)
- **Branch:** main
- **Se ainda servindo código antigo:** Variables → adicionar `NO_CACHE=1` → Redeploy (ou Clear Build Cache no deploy)

## Dockerfile (desde este commit)

O `dashboard/Dockerfile` força build determinístico e ignora cache do Nixpacks. Railway usa o Dockerfile automaticamente quando presente na raiz do serviço (dashboard/).

## Commit Hashes (origin/main)

| Feature | Commit |
|---------|--------|
| startingCapital 500 / 5000 | 47638a2 |
| /api/deployment-info | f306cc8 |
| /api/shadow/audit | 66d3808 |
| /api/version | (this deploy) |

## File Paths (must exist in deployed build)

- `dashboard/app/api/deployment-info/route.ts`
- `dashboard/app/api/shadow/audit/route.ts`
- `dashboard/app/api/version/route.ts`
- `dashboard/lib/shadowSimulationProfiles.ts` (startingCapital 500, 5000)
- `dashboard/railway.json`

## Public URLs to Test

Base URL: `https://dashboard-next-production-b126.up.railway.app`

| URL | Expected Status | Expected Response |
|-----|-----------------|--------------------|
| `/api/version` | 200 | `{"success":true,"gitCommitSha":"...","routeExists":true,...}` |
| `/api/deployment-info` | 200 | `{"success":true,"dashboardUrl":"https://...","hasRailwayDomain":true}` |
| `/api/shadow/profiles` | 200 | `startingCapital: 500` (shadow_100), `5000` (shadow_1000) |
| `/api/shadow/audit` | 200 | `{"negativeExpectancy",...,"profileSummaries",...}` |

## Deterministic Verification Order

1. **GET /api/version** — If 404: old code. If 200: check `gitCommitSha` matches latest main.
2. **GET /api/deployment-info** — If 404: pre-f306cc8. If 200: deployment-info deployed.
3. **GET /api/shadow/profiles** — Check `startingCapital` is 500/5000. If 100/1000: pre-47638a2.
4. **GET /api/shadow/audit** — If 404: pre-66d3808. If 200: full audit available.

## Stale Deploy Detection

If `/api/version` returns 404 → deployment is old (version route added in this refresh).
If `/api/version` returns 200 but `gitCommitSha` does not match `git rev-parse origin/main` → Railway built from wrong commit or cache.
