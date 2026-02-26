# Teste Formal de Determinismo do Seed Global

## Objetivo

Validar que `set_global_seed(seed)` torna o ambiente determinístico para:
- `random`
- `numpy`
- `hash()`
- `torch` (se disponível)

---

## 1. Instruções de Execução

### Pré-requisito: PYTHONHASHSEED

**PYTHONHASHSEED deve ser setado ANTES do processo Python iniciar.**  
Setar `os.environ["PYTHONHASHSEED"]` dentro do script é **tarde demais** — o salt do `hash()` já foi inicializado na startup do interpretador.

```bash
# Teste 1: Mesmo seed, dois runs — devem ser idênticos
PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_seed_determinism.py > run1.json
PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_seed_determinism.py > run2.json
diff run1.json run2.json
```

**Esperado:** Nenhuma diferença (exit code 0, saída vazia).

```bash
# Teste 2: Seed diferente — deve divergir
PYTHONHASHSEED=43 RANDOM_SEED=43 python scripts/test_seed_determinism.py > run3.json
diff run1.json run3.json
```

**Esperado:** Diferenças detectadas (exit code 1, diff mostrando mudanças).

---

## 2. Validação de PYTHONHASHSEED

O script inclui `pythonhashseed_set_before_launch` no JSON. Ele indica se `PYTHONHASHSEED` estava em `os.environ` **antes** de chamar `set_global_seed()`.

### Como verificar

```bash
PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_seed_determinism.py | grep pythonhashseed_set_before_launch
# Deve mostrar: "pythonhashseed_set_before_launch": true
```

```bash
RANDOM_SEED=42 python scripts/test_seed_determinism.py | grep pythonhashseed_set_before_launch
# Mostra: "pythonhashseed_set_before_launch": false
```

### Por que falha se não estiver setado antes do processo iniciar?

O Python 3.3+ usa um salt aleatório para `hash()` na inicialização do interpretador. Esse salt é fixado **antes** de qualquer código Python rodar. Se `PYTHONHASHSEED` não estiver no ambiente ao iniciar o processo:

- Cada execução usa um salt diferente
- `hash("arbitrage_test")` varia entre runs mesmo com o mesmo `RANDOM_SEED`
- `run1.json` e `run2.json` divergirão no campo `"hash"`

Setar `os.environ["PYTHONHASHSEED"]` dentro do script só afeta **subprocessos** criados depois disso, não o processo atual.

---

## 3. Critério Formal de Aprovação

**Seed determinístico aprovado se e somente se:**

| Condição | Comando | Esperado |
|----------|---------|----------|
| run1 == run2 byte a byte | `diff run1.json run2.json` | Exit 0, saída vazia |
| run1 != run3 | `diff run1.json run3.json` | Exit 1, diferenças mostradas |
| hash idêntico entre runs com mesmo seed | Comparar `"hash"` em run1 e run2 | Valores iguais |
| torch idêntico se disponível | Comparar `"torch"` em run1 e run2 | Valores iguais ou ambos `null` |
| pythonhashseed_set_before_launch | run1 e run2 | `true` |

**Se qualquer divergência ocorrer**, possíveis causas:

- **import antes do seed** — algum módulo usou `random`/`hash` antes de `set_global_seed()`
- **múltiplos workers** — cada worker é um processo com estado próprio
- **uso de SystemRandom** — não é afetado por `random.seed()`
- **geração paralela** — threads/processos sem seed compartilhado
- **PYTHONHASHSEED não setado na launch** — `hash()` não determinístico

---

## 4. Estrutura da Saída JSON

```json
{
  "seed": 42,
  "pythonhashseed_set_before_launch": true,
  "random": [0.6394267984578837, ...],
  "numpy": [0.3745401188473625, ...],
  "hash": "1234567890123456789",
  "torch": [0.4963, ...]
}
```

- `numpy` e `torch` são `null` se as bibliotecas não estiverem instaladas.

---

## 5. Conclusão

Quando todos os testes acima passarem, o laboratório pode ser considerado **deterministicamente controlado**.
