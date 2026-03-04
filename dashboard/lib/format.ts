export function pct(value: number | undefined | null, decimals = 0): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function bps(value: number | undefined | null, decimals = 1): string {
  if (value == null) return "—";
  return `${value.toFixed(decimals)} bps`;
}

export function num(value: number | undefined | null, decimals = 1): string {
  if (value == null) return "—";
  return value.toFixed(decimals);
}

export function regimeLabel(state: string | undefined): string {
  switch (state) {
    case "high_edge":
      return "ALTA OPORTUNIDADE";
    case "neutral":
      return "NEUTRO";
    case "hostile":
      return "DESFAVORÁVEL";
    default:
      return "CARREGANDO...";
  }
}

export function regimeColor(state: string | undefined): string {
  switch (state) {
    case "high_edge":
      return "text-terminal-green terminal-glow-green";
    case "neutral":
      return "text-terminal-yellow terminal-glow-yellow";
    case "hostile":
      return "text-terminal-red terminal-glow-red";
    default:
      return "text-terminal-muted";
  }
}

export function regimeDot(state: string | undefined): string {
  switch (state) {
    case "high_edge":
      return "pulse-dot pulse-dot-green";
    case "neutral":
      return "pulse-dot pulse-dot-yellow";
    case "hostile":
      return "pulse-dot pulse-dot-red";
    default:
      return "pulse-dot";
  }
}

export function qualityLevel(value: number, thresholdHigh = 0.7, thresholdMid = 0.4): string {
  if (value >= thresholdHigh) return "Alta";
  if (value >= thresholdMid) return "Média";
  return "Baixa";
}

export function qualityColor(value: number, thresholdHigh = 0.7, thresholdMid = 0.4): string {
  if (value >= thresholdHigh) return "text-terminal-green";
  if (value >= thresholdMid) return "text-terminal-yellow";
  return "text-terminal-red";
}

export function truncateId(id: string, len = 8): string {
  if (!id) return "—";
  return id.length > len ? id.slice(0, len) + "…" : id;
}
