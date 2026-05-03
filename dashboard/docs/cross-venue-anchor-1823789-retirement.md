/**
 * Retirement note — crossVenueAnchor1823789
 *
 * Formal retirement of Gamma market **1823789** como *economic proof track* vivo.
 */

## Identificação

| Campo | Valor |
| ----- | ----- |
| **marketId** | `1823789` |
| **Pergunta** | Will Ethereum reach $4,000 in April? |
| **Data da decisão** | 2026-05-03 |
| **Decisão** | **Retired as economic proof track** — não usar como evidência económica viva. |

## Motivo principal

Mercado **temporalmente vencido** em maio relativamente ao evento nominal (April), com sintomas estruturais de ilíquidez/markouts não informativos.

## Sintomas observados

- `markoutInformativenessVerdict` = **`WEAK_FLAT_MARKOUTS`**
- **8** ciclos fechados via markout; **0** informativos
- **24** `okFollowups`, todos com **markout zero**
- `best_ask_only` como **única** fonte de preço recuperável nos followups
- `lowPricePinnedCycleCount` = **8**
- Preço efectivo ~**0.001** (“pinned” observacional)
- `clob_book_unavailable` recorrente no pipeline paper/shadow observado

## Regras aprendidas

1. Mercado **vencido**, **fec** **resolvido** ou **estruturalmente degenerado em micro-preço/livro** não pode ser premissa para **prova económica viva**.
2. **Market suitability** (read-only) vem **antes** de worker, watcher, readiness económico vivente ou discussão microcapital — *mercado primeiro, código depois*.
