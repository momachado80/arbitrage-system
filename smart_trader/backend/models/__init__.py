try:
    from models.signal import (
        Signal, Evento, WhaleActivity, MercadoPolymarket,
        Recomendacao, AnaliseDetalhada, HistoricoCategoria,
        Categoria, Urgencia, CATEGORIA_MULTIPLIERS, CATEGORIA_KEYWORDS
    )
except ImportError:
    from .signal import (
        Signal, Evento, WhaleActivity, MercadoPolymarket,
        Recomendacao, AnaliseDetalhada, HistoricoCategoria,
        Categoria, Urgencia, CATEGORIA_MULTIPLIERS, CATEGORIA_KEYWORDS
    )

__all__ = [
    "Signal", "Evento", "WhaleActivity", "MercadoPolymarket",
    "Recomendacao", "AnaliseDetalhada", "HistoricoCategoria",
    "Categoria", "Urgencia", "CATEGORIA_MULTIPLIERS", "CATEGORIA_KEYWORDS"
]

