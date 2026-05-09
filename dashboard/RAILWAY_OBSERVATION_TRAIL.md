# Trilha de observação contínua no Railway (arbitrage-system)

Este serviço já usa `node server.js` (ver `railway.json` e `Dockerfile`). O Next.js corre em produção com `instrumentation.ts`, que arranca os schedulers dos probes no boot. O `GET /api/healthz` garante os mesmos schedulers se o boot falhar por algum caminho alternativo.

## Volume persistente (recomendado)

1. Railway → serviço **arbitrage-system** → **Volumes** → Add volume.  
2. **Mount path:** `/data`  
3. Variáveis (ver secção seguinte): `PAPER_STATE_DIR=/data`  
4. Se já usas shadow persistence no mesmo volume, mantém também `SHADOW_PERSISTENCE_PATH=/data` (ou subpastas distintas se preferires organizar ficheiros manualmente).

Com `PAPER_STATE_DIR=/data`, os ficheiros da trilha ficam **directamente** em `/data`:

- `pocket-economics-state.json`
- `pocket-execution-state.json`
- `minimal-paper-execution-state.json`
- `system-ladder-history.json`

Sem volume e sem `PAPER_STATE_DIR`, o default continua a ser `<cwd>/.paper/` (filesystem efémero no deploy — estado perde-se entre deploys).

## Variáveis de ambiente recomendadas

| Variável | Valor sugerido | Notas |
|----------|----------------|--------|
| `NODE_ENV` | `production` | Normalmente definido pelo Railway. |
| `PORT` | (injectado pelo Railway) | Não sobrescrever manualmente. |
| `PAPER_STATE_DIR` | `/data` | **Obrigatório para observação longa** com volume montado em `/data`. |
| `SHADOW_PERSISTENCE_PATH` | `/data` | Se usas shadow; ver `RAILWAY_SHADOW_PERSISTENCE.md`. |

Opcionais (só se precisares de caminhos absolutos por ficheiro, sobrepondo `PAPER_STATE_DIR`):

- `POCKET_ECON_STATE_PATH`
- `POCKET_EXEC_STATE_PATH`
- `MINIMAL_PAPER_STATE_PATH`
- `SYSTEM_LADDER_HISTORY_PATH`

Desligar escrita em disco (estado só em memória):

- `POCKET_ECON_DISABLE_DISK=1`
- `POCKET_EXEC_DISABLE_DISK=1`
- `MINIMAL_PAPER_DISABLE_DISK=1`
- `SYSTEM_LADDER_HISTORY_DISABLE_DISK=1`

## Start command recomendado

- **Railway / Dockerfile:** `node server.js` (já definido em `railway.json` → `deploy.startCommand` e `Dockerfile` `CMD`).  
- Equivale a `npm run start` em espírito (build + Next produção), mas o `server.js` faz bind explícito a `0.0.0.0` e usa `PORT`.

Não é necessário alterar o start command para a trilha de probes.

## Checklist pós-deploy

1. Build no Railway conclui sem erros.  
2. Serviço fica **Active** e responde na URL pública.  
3. Volume montado em `/data` e `PAPER_STATE_DIR=/data` definido.  
4. Primeiro `GET /api/healthz` mostra schedulers dos probes a correr e `persistencePath` apontando para `/data/...` onde aplicável.  
5. Após um redeploy, estados em `/data` mantêm-se (reidratação visível nos endpoints).  
6. `ladderHistory.snapshotCount` em `/api/probe/system-ladder` aumenta ou mantém-se conforme mudanças agregadas.

## Curls (substituir `BASE` pela URL pública, ex. `https://<projeto>.up.railway.app`)

```bash
BASE="https://SEU_DOMINIO_RAILWAY"

curl -sS "$BASE/api/healthz" | head -c 4000

curl -sS "$BASE/api/probe/system-ladder" | head -c 6000

curl -sS "$BASE/api/probe/catalog-pocket" | head -c 2000

curl -sS "$BASE/api/probe/pocket-economics" | head -c 2000

curl -sS "$BASE/api/probe/pocket-execution" | head -c 2000

curl -sS "$BASE/api/probe/minimal-paper-execution" | head -c 2000
```

CLI local contra produção:

```bash
LADDER_STATUS_URL="$BASE" npm run ladder-status
```

## Implementação no código

- `lib/paperStateDir.ts` — `PAPER_STATE_DIR` e nomes dos ficheiros (`PAPER_TRAIL_FILENAMES`).  
- Defaults de persistência em pocket-economics, pocket-execution, minimal-paper-execution, system-ladder-history e hygiene paths alinham-se com este módulo.

Nenhuma alteração a thresholds, gates, execution engine ou lógica económica dos probes.
