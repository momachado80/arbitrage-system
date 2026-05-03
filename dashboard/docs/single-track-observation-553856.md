# Single Track Observation Dossier — 553856 Thunder Finals

- **marketId:** `553856`
- **status:** paper/shadow observation candidate only (sem execução real, sem ordens, sem carteira nem credenciais de assinatura)
- **decisão humana:** selected for first single-track observation (**observação** paper/shadow apenas)

## Motivo da escolha (read-only readiness)

| Sinal |
| :--- |
| `resolutionClarityScore` = **5** |
| `observabilityValueScore` = **4** |
| `ambiguityRisk` = **LOW** |
| `eventShockRisk` = **MEDIUM** |
| `mid` informativo (livro próximo ao meio probabilístico) |
| Livro **two-sided** quando observado pelo CLOB |
| `endDate` futura e explícita (horizonte conhecido) |

## Contexto rápido frente aos descartados

- **`540844`** (bitcoin vs GTA VI): maior observabilidade mecânica histórica, mas **ambiguidade** média sobre oráculos de preço/acoplamento temporal e **`eventShockRisk` alto** — menos limpo como *primeira* régua quando o objectivo é isolar comportamento bilateral previsível.
- **`558934`** / FIFA: excelente clareza de resolução desportiva, mas **`eventShockRisk` bem mais alto** (janelas de notícias de selecções, convocatórias, lesões massivas que podem atropelar microestrutura durante horas inteiras).
- **`1823789` retired:** este mercado existe apenas como Âncora legada/readiness transversal — **nunca** pode servir como *economic proof track* para novas decisões nem como substituto de observação económica viva dos candidatos Gamma.

## Regra de promoção (“mercado primeiro”)

- O mercado **só avança** no pipeline económico se produzir **markout informativo** repetível sob **paper/shadow** auditável (`paper_cycle_positive` quando essa ferramenta estiver disponível ao track).
- `canUseForMicrocapitalCandidate` **permanece `false`** em todas as fases actuais; qualquer uso futuro seria porta separada explicitamente autorizada por humanos.
- Qualquer sinal repetido de **`flat`** / **`price pinned`** / **livro inexistente-only-one-side** deve **bloquear continuidade** automática até revisão manual.

---

## Critérios de morte (observação encerra ou congela o track)

- Livro CLOB **indisponível** de forma **recorrente** (timeouts, HTTP instável, payloads vazios).
- **`price pinned`** (degraus ~0/~1 probabilísticos) ou marcação bloqueante no gate de suitability.
- **Apenas `bestAsk` ou apenas `bestBid`** de forma **recorrente** (livro efectivamente uni-lateral).
- **Markouts flat** documentados nos horizontes de prova já adoptados (**5 s / 30 s / 60 s**) sem exploração informativa.
- **`priceUnavailableRate` elevado** (fraccão alta de ciclo onde não há quotização fiável sobre o token YES observado).
- **Ausência** de ciclo/paper evidence positivo (**`paper_cycle_positive`**) após janela de calor inicial definida no diário técnico.
- **Nova ambiguidade** na declaração de resolução (mudanças de texto, debates de arbitragem política‑legal não previstos, incerteza institucional explícita na fonte oficial).

---

## Critérios de continuidade (track permanece válido)

- Livro **two-sided** durante a maior parte dos ciclos de amostragem.
- Séries de **`followUps` 5 s / 30 s / 60 s** colectadas para os mesmos identificadores (continuidade de prova longitudinal).
- **Markouts informativos** (mudanças de marca documentáveis dentro da banda económica de interesse observacional — sem prometer monetização nem edge).
- **Preços não pressos contra o corrimão** probabilístico (sem colagem permanente aos extremos).
- **Índices baixos de `unavailable` / erro** relativamente aos ticks observados naquele período — coerência de dados antes de storytelling.
- **Logs auditáveis** (checksum e metadados mínimos, sem payloads brutos inteiros) para cada ciclo registado pelo stack paper/shadow.

---

> Este dossier não altera parâmetros de execução, thresholds económicos, `dynamic exit`, nem estratégia. É um documento vivo de readiness para **humano + stack read-only/paper**.
