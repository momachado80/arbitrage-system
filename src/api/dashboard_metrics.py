"""
Dashboard Metrics — Métricas e Textos Humanos
===============================================

Responsável por:
- human_status (OK | Atenção | Modo Seguro)
- status_color (green | yellow | red)
- Tradução de streaks
- Nível de risco
- explanation_text inteligente

Não modifica core. Apenas deriva informação para exibição.
"""

from typing import Any, Dict, Optional, Tuple

# Mapa de reasons técnicas para texto humano
HUMAN_REASON_MAP: Dict[str, str] = {
    "RECONCILE_STALE": "O robô não conseguiu confirmar o saldo recentemente.",
    "TRADE_DROP": "Uma confirmação de execução pode ter sido perdida.",
    "CAPITAL_MISMATCH": "O saldo interno diverge do saldo da corretora.",
    "INVARIANT_FAIL": "Foi detectada inconsistência contábil.",
    "THREAD_DEAD": "Um componente interno parou inesperadamente.",
    "EXCHANGE_UNREACHABLE": "A corretora não está respondendo.",
    "SAFE_MODE": "Modo de segurança ativado automaticamente.",
    "CONFIG_CHANGED": "As configurações foram alteradas.",
    "CLOCK_ANOMALY": "O relógio do sistema apresentou comportamento inesperado.",
}


def get_human_status(safety_state: Any) -> str:
    """
    Deriva status humano: OK | Atenção | Modo Seguro.
    """
    if safety_state is None:
        return "OK"
    if safety_state.global_safe_mode:
        return "Modo Seguro"
    if safety_state.buy_blocked or safety_state.reasons:
        return "Atenção"
    return "OK"


def get_status_color(safety_state: Any) -> str:
    """
    Mapeia cor para o indicador: green | yellow | red.
    """
    if safety_state is None:
        return "green"
    if safety_state.global_safe_mode:
        return "red"
    if safety_state.buy_blocked or safety_state.reasons:
        return "yellow"
    return "green"


def translate_streak(streak_name: str, value: int) -> str:
    """
    Traduz streak para texto humano.
    """
    if value >= 3:
        return f"{streak_name}: OK ({value} confirmações consecutivas)"
    return f"{streak_name}: aguardando ({value}/3)"


def get_risk_level(safety_state: Any) -> str:
    """
    Nível de risco: low | medium | high | critical.
    """
    if safety_state is None:
        return "low"
    if safety_state.global_safe_mode:
        return "critical"
    reasons = safety_state.reasons
    if not reasons:
        return "low"
    critical_reasons = {"CAPITAL_MISMATCH", "INVARIANT_FAIL", "THREAD_DEAD"}
    if any(r in critical_reasons for r in reasons):
        return "high"
    return "medium"


def build_explanation_text(
    safety_state: Any,
    app_config: Any,
    position_manager: Any = None,
    capital_manager: Any = None,
    exposure_percent: float = 0.0,
) -> str:
    """
    Gera texto explicativo inteligente e contextual para o operador.

    Narrativa melhorada:
    - Saudável: "Monitorando mercados normalmente."
    - Saudável sem posições: "Nenhuma posição aberta no momento."
    - buy_blocked: "Compras pausadas temporariamente por segurança."
    - SAFE_MODE: "Modo de segurança ativado automaticamente."
    - Exposição > 50%: "Nível de exposição elevado."
    """
    has_positions = False
    if position_manager:
        try:
            all_pos = position_manager.get_all_positions()
            has_positions = any(
                getattr(p, "shares", 0) > 0 for p in all_pos.values()
            )
        except Exception:
            pass

    if safety_state is None:
        return "Monitorando mercados normalmente." if has_positions else "Nenhuma posição aberta no momento."

    reasons = safety_state.reasons
    if not reasons:
        base = "Monitorando mercados normalmente."
        if not has_positions:
            base = "Nenhuma posição aberta no momento."
        if exposure_percent > 50 and has_positions:
            base += " Nível de exposição elevado."
        return base

    if safety_state.global_safe_mode:
        return "Modo de segurança ativado automaticamente. Aguarde confirmação manual para retomar."

    if safety_state.buy_blocked:
        return "Compras pausadas temporariamente por segurança."

    if "EXCHANGE_UNREACHABLE" in reasons:
        return "A corretora não está respondendo. Aguardando reconexão."

    if "RECONCILE_STALE" in reasons:
        return "O robô não conseguiu confirmar o saldo recentemente. Aguardando reconciliação."

    if "TRADE_DROP" in reasons:
        return "Uma confirmação de execução pode ter sido perdida. Verificando pendências."

    if "CONFIG_CHANGED" in reasons:
        return "As configurações foram alteradas. Valide e confirme para continuar."

    if "CAPITAL_MISMATCH" in reasons:
        return "O saldo interno diverge do saldo da corretora. Verificação necessária."

    if "INVARIANT_FAIL" in reasons:
        return "Foi detectada inconsistência contábil. Aguardando confirmações consecutivas."

    if "THREAD_DEAD" in reasons:
        return "Um componente interno parou inesperadamente. Reinicie o sistema."

    return "O sistema está em estado de atenção. Verifique os alertas abaixo."


def apply_narrative_alignment(
    base_explanation: str,
    confidence: Dict[str, Any],
    financial_risk: Dict[str, Any],
) -> str:
    """
    Harmoniza narrativa quando Confidence e Risk divergem.

    - Confidence alta (>=80) + Risco Elevado: adiciona "Sistema operacionalmente
      saudável, porém com exposição financeira elevada."
    - Confidence baixa + Risco Baixo: "Risco financeiro baixo, mas há problemas
      operacionais a resolver."
    """
    conf_score = confidence.get("score") or 0
    conf_level = confidence.get("level") or ""
    risk_level = financial_risk.get("level") or ""

    if conf_score >= 80 and risk_level == "Elevado":
        return base_explanation.rstrip(". ") + ". Sistema operacionalmente saudável, porém com exposição financeira elevada."
    if conf_score < 50 and risk_level == "Baixo":
        return base_explanation.rstrip(". ") + ". Risco financeiro baixo, mas há problemas operacionais a resolver."
    return base_explanation


def humanize_reasons(reasons: set) -> list:
    """Retorna lista de textos humanos para as reasons, ordenada."""
    return [HUMAN_REASON_MAP.get(r, r) for r in sorted(reasons)]


def compute_confidence_score(system_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula indicador de confiança 0-100.

    Subtrai:
    - 30 se global_safe_mode
    - 20 se buy_blocked
    - 10 se last_reconcile_age > reconcile_stale_seconds / 2
    - 10 se invariants_ok_streak < 3
    - 10 se capital_mismatch_ok_streak < 3
    - 20 se reasons ativas (não contabilizado se já safe_mode ou buy_blocked)

    Nível: >= 80 Alta, >= 50 Moderada, < 50 Baixa
    """
    score = 100
    safety_state = system_context.get("safety_state")
    app_config = system_context.get("app_config") or system_context.get("config")

    if safety_state is None:
        return {"score": 100, "level": "Alta", "color": "green"}

    reconcile_stale = (
        getattr(app_config, "reconcile_stale_seconds", 60.0) if app_config else 60.0
    )
    half_stale = reconcile_stale / 2.0

    if safety_state.global_safe_mode:
        score -= 30
    if safety_state.buy_blocked or safety_state.reasons:
        score -= 20

    rec_ts = safety_state.last_reconcile_ok_ts
    if rec_ts is not None:
        import time
        age = time.time() - rec_ts
        if age > half_stale:
            score -= 10

    if safety_state.invariants_ok_streak < 3:
        score -= 10
    if safety_state.capital_mismatch_ok_streak < 3:
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        level, color = "Alta", "green"
    elif score >= 50:
        level, color = "Moderada", "yellow"
    else:
        level, color = "Baixa", "red"

    return {"score": score, "level": level, "color": color}


def compute_financial_risk_from_inputs(
    capital_data: Dict[str, Any],
    positions_count: int,
    safety_state: Any,
    max_concentration_ratio: float,
) -> Dict[str, Any]:
    """
    Usa risk_model (gradual) para risco financeiro.
    Retorna: level, color, score, description.
    """
    from src.api.risk_model import (
        RISK_MODEL_VERSION,
        RiskInputs,
        build_risk_explanation,
        compute_risk_score,
        risk_level_from_score,
    )

    total = capital_data.get("total_capital") or 0.0
    locked = (
        (capital_data.get("locked_in_open_orders") or 0.0)
        + (capital_data.get("locked_in_unresolved_markets") or 0.0)
    )
    percent_reserved = (locked / total) if total > 0 else 0.0

    inputs = RiskInputs(
        percent_reserved=min(1.0, percent_reserved),
        max_concentration=min(1.0, max_concentration_ratio),
        open_positions=max(0, positions_count),
        global_safe_mode=bool(safety_state and safety_state.global_safe_mode),
    )
    score = compute_risk_score(inputs)
    result = risk_level_from_score(score)
    result["description"] = build_risk_explanation(inputs, score)
    result["model_version"] = RISK_MODEL_VERSION
    result["percent_reserved"] = inputs.percent_reserved
    result["max_concentration"] = inputs.max_concentration
    result["open_positions"] = inputs.open_positions
    result["global_safe_mode"] = inputs.global_safe_mode
    return result


def compute_concentration(
    exposure_summary: list,
    total_capital: float,
) -> Dict[str, Any]:
    """
    Retorna maior exposição por mercado.

    Se > 40% → elevated: True (destacar no UI).
    """
    if not exposure_summary or total_capital <= 0:
        return {
            "market": None,
            "percent": 0.0,
            "exposure": 0.0,
            "elevated": False,
            "text": "Nenhuma exposição.",
        }

    top = max(exposure_summary, key=lambda e: e.get("percent", 0))
    pct = top.get("percent", 0.0)
    market = top.get("market", "")
    exp = top.get("exposure", 0.0)

    return {
        "market": market,
        "percent": round(pct, 1),
        "exposure": round(exp, 2),
        "elevated": pct > 40,
        "text": f"Maior exposição: {market} ({pct:.1f}% do capital)",
    }


def compute_drawdown_from_tracker(
    drawdown_tracker: Any,
    current_capital: float,
) -> Dict[str, Any]:
    """
    Usa DrawdownTracker (persistente) para drawdown.
    Fallback se tracker ausente.
    """
    if drawdown_tracker is not None and hasattr(drawdown_tracker, "get_drawdown"):
        return drawdown_tracker.get_drawdown(current_capital or 0.0)
    return {
        "capital_max": current_capital or 0.0,
        "capital_current": current_capital or 0.0,
        "drawdown_percent": 0.0,
        "text": "—",
    }


# Retrocompatibilidade: compute_drawdown em memória (para testes)
def compute_drawdown(
    capital_history: list,
    current_capital: float,
) -> Dict[str, Any]:
    """Drawdown em memória (legado). Preferir DrawdownTracker."""
    if not capital_history or current_capital is None:
        return {
            "capital_max": current_capital or 0.0,
            "capital_current": current_capital or 0.0,
            "drawdown_percent": 0.0,
            "text": "—",
        }
    capitals = [h.get("total_capital") for h in capital_history if h.get("total_capital") is not None]
    capital_max = max(capitals) if capitals else current_capital
    if capital_max <= 0:
        return {
            "capital_max": current_capital or 0.0,
            "capital_current": current_capital or 0.0,
            "drawdown_percent": 0.0,
            "text": "Sem histórico.",
        }
    dd_pct = max(0.0, (capital_max - current_capital) / capital_max * 100)
    text = f"Drawdown atual: {dd_pct:.1f}%" if dd_pct > 0 else "No máximo histórico."
    return {
        "capital_max": round(capital_max, 2),
        "capital_current": round(current_capital, 2),
        "drawdown_percent": round(dd_pct, 1),
        "text": text,
    }
