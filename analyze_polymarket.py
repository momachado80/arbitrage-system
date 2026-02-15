#!/usr/bin/env python3
"""
Script de Análise de Arbitragem - Polymarket
=============================================

Executa análise completa de oportunidades de arbitragem no Polymarket
baseado em divergências de probabilidade.

Uso:
    python analyze_polymarket.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.engine.arbitrage_analyst import ArbitrageAnalyst

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Função principal."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ANALISADOR DE ARBITRAGEM - POLYMARKET                                ║
║         Especialista em Previsão de Mercados                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Este analisador identifica oportunidades de arbitragem baseadas em:        ║
║  • Divergências de probabilidade entre preço e expectativa                   ║
║  • Volume e liquidez suficientes (> $100.000)                                ║
║  • Timeframes curtos (< 48 horas)                                           ║
║  • Expected Value (EV) positivo                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        async with ArbitrageAnalyst() as analyst:
            print("\n🔍 Escaneando mercados do Polymarket em tempo real...")
            print("\n   Critérios de Filtragem:")
            print(f"   ✓ Volume mínimo: ${analyst.MIN_VOLUME_USD:,.0f}")
            print(f"   ✓ Timeframe máximo: {analyst.MAX_TIMEFRAME_HOURS} horas")
            print(f"   ✓ Divergência mínima: {analyst.MIN_DIVERGENCE*100:.0f}%")
            print(f"   ✓ EV mínimo: ${analyst.MIN_EV:.2f} por $1 investido")
            print("\n   Aguarde... Isso pode levar alguns minutos...\n")
            
            opportunities = await analyst.scan_opportunities(limit=200)
            
            # Exibir resultados
            summary = analyst.format_summary(opportunities)
            print(summary)
            
            # Informações adicionais
            if opportunities:
                print("\n" + "="*80)
                print("INFORMAÇÕES IMPORTANTES:")
                print("="*80)
                print("""
⚠️  AVISOS LEGAIS E DE RISCO:

1. Este é um sistema de análise automatizada. Sempre faça sua própria pesquisa
   antes de tomar decisões de investimento.

2. As probabilidades esperadas são estimadas usando heurísticas. Em produção,
   você deveria integrar com:
   - Odds de casas de apostas tradicionais
   - Modelos de previsão especializados
   - Agregadores de odds confiáveis

3. Arbitragem de mercados de previsão envolve riscos:
   - Risco de base (mercados podem não ser perfeitamente equivalentes)
   - Risco de liquidez (ordens podem não executar ao preço esperado)
   - Risco de resolução (disputas sobre resultados)
   - Risco de taxas e custos de transação

4. Sempre teste estratégias em modo paper trading antes de usar capital real.

5. Este sistema é fornecido "como está", sem garantias de lucro ou precisão.
                """)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Análise interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro durante análise: {e}", exc_info=True)
        print(f"\n❌ Erro durante análise: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())




