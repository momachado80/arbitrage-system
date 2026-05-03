# Aposentadoria do track `crossVenueAnchor1823789` (mercado 1823789)

## Mercado

**Pergunta:** Will Ethereum reach $4,000 in April?  
**Gamma market id:** `1823789`

## Motivo

Mercado temporalmente **expirado** em contexto económico vivo: referência abril 2026, com avaliação a **3 de maio de 2026** já fora da janela do evento. Não deve servir como prova de readiness **económica atual**.

## Sintomas observados

- **markoutInformativenessVerdict** = `WEAK_FLAT_MARKOUTS`
- **informativeMarkoutCycleCount** = `0`; **flatMarkoutCycleCount** = `8`
- **lowPricePinnedCycleCount** = `8`; **bestAskOnlyCycleCount** = `8`
- Followups marcados como ok com **basePrice/followupPrice** presos em **0.001** e **markouts zero**
- **`clob_book_unavailable`** recorrente no worker / gate narrow

## Decisão operacional

- **Não** usar `1823789` como evidência para microcapital económico vivo.
- Manter apenas como **caso histórico** ou fixture de pipeline / testes, se necessário.
- Novos candidatos a candidatura microcapital devem passar pela **Market Suitability Gate** (read-only) antes de serem considerados.

## Regra aprendida

Mercado temporalmente **vencido**, **resolve** ou **estruturalmente ilíquido / preso a tick mínimo** não pode ser premissa para **readiness económica viva**.
