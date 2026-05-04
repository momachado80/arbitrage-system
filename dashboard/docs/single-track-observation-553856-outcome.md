# Observação single-track — aposentadoria do primeiro candidato (553856)

- **marketId:** `553856`
- **Decisão:** aposentado como primeiro candidato de observação paper/shadow (**retired as first single-track observation candidate**).

## Motivos

- **5 / 5** amostras com veredito `FLAT_SAMPLE`
- **`informativeSamples` = 0**
- Markouts aos horizontes **5s / 30s / 60s** = **0** em todas as amostras
- **`midRange` = 0** (precificação média estável durante a sessão observada)
- Livro bilateral (two-sided) na prática avaliável, mas **sem variação mid informativa** na bateria definida

## Conclusão

- Mercado **útil para validar o sampler read-only** (Gamma em path `/markets/<id>`, snapshots t0+t*, digest de markouts)
- **Não adequado** como primeiro **campo de prova econômica** (ausência de sinal explorável só com esse protocolo conservador read-only)

## Regra aplicada

- **`5`** amostras consecutivas com **`FLAT_SAMPLE`** → encerramento do track como primeira faixa observacional (`5 FLAT_SAMPLE encerram o track`)

## Uso recomendado

- **Não usar** este mercado como base para **microcapital** sem outro desenho de prova (`canUseForMicrocapitalCandidate` permanece **`false`**)
