# Fase 1.5 — Validação de Determinismo

**Objetivo:** Provar que o ambiente é reproduzível. Não assumir. Provar.

---

## 1️⃣ Congelamento Real de Dependências

### Procedimento com pip-tools (recomendado)

```bash
# Usar Python 3.11
python3.11 -m venv venv_clean
source venv_clean/bin/activate
pip install --upgrade pip
pip install pip-tools
pip-compile requirements.in --output-file=requirements.txt
```

Ou execute o script:

```bash
./scripts/freeze_requirements.sh
```

### Regenerar após alterar requirements.in

```bash
pip-compile requirements.in --output-file=requirements.txt
```

**Commit** o `requirements.txt` após qualquer alteração.

---

## 2️⃣ Seed Global

Implementado em `src/server.py`:

- `_set_global_seed(seed)` — chamada antes de qualquer import de engine
- `PYTHONHASHSEED` definido
- `random.seed()`, `numpy.random.seed()`, `torch.manual_seed()` (se disponíveis)
- Variável de ambiente: `RANDOM_SEED` (default: 42)
- Log: `GLOBAL_SEED_SET: 42`

---

## 3️⃣ Boot Determinístico — Checklist Formal

### Executar localmente

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Validar

| # | Verificação | Esperado |
|---|-------------|----------|
| 1 | Log de versão | `PYTHON_VERSION: 3.11.x` |
| 2 | Log de seed | `GLOBAL_SEED_SET: 42` |
| 3 | Log de env | `ENV: production` (ou development) |
| 4 | Health responde | `curl http://localhost:8000/health` → `{"status":"ok",...}` |
| 5 | Sem warnings | `python -Wd -m uvicorn src.server:app` — zero warnings |

### Comando para forçar warnings

```bash
python -Wd -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

---

## 4️⃣ Teste de Reprodutibilidade

**Procedimento:**

1. Rodar simulação por 10 minutos (RUN_MODE=REALISTIC_SIMULATION)
2. Parar
3. Reiniciar (mesmo seed, mesmos dados)
4. Rodar novamente 10 minutos

**Comparar:**

- Número de trades
- PnL final
- Slippage médio
- Rejection rate

**Resultado esperado:** Valores idênticos se seed fixa e dados forem os mesmos.

**Se não coincidirem:** Existe componente estocástico não controlado.

---

## 5️⃣ Teste no Render — Validação Cruzada

Após deploy, verificar nos logs:

- [ ] Python 3.11 confirmado
- [ ] `GLOBAL_SEED_SET` aparece
- [ ] Sem stack trace
- [ ] `/health` responde

**Se versão Python não for 3.11.x:** Render pode estar ignorando `runtime.txt`. Verificar configuração do serviço.

---

## 6️⃣ Hardening Implementado

| Item | Implementação |
|------|---------------|
| Workers | `--workers 1` no Procfile |
| Fail fast | Rejeita Python < 3.11 ou >= 3.12 |
| Git commit | `GIT_COMMIT: <hash>` nos logs de boot |

---

## Script de Validação

```bash
./scripts/validate_boot.sh
```

---

## Próximo Marco

Quando tudo acima estiver validado:

**Ambiente experimental controlado.**

Só então: **Fase 2 — Validação de Ingestão Quantitativa.**
