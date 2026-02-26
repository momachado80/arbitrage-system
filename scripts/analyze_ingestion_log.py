#!/usr/bin/env python3
"""
Analisa log de ingestão e emite veredicto PASS/FAIL.

Uso:
    python scripts/analyze_ingestion_log.py logs/observe.log

Critérios de aprovação:
- snapshots_window nunca zero
- engine_events_window nunca zero
- discard_rate_window < 0.02
- silence_seconds < 30
- ws_disconnects_total <= 1
"""

import json
import sys
from pathlib import Path


def extract_audit_lines(path: Path) -> list[dict]:
    """Extrai linhas INGESTION_AUDIT do log."""
    audits = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "INGESTION_AUDIT" not in line or '"kind"' not in line:
            continue
        # Extrair JSON: do primeiro { ao último }
        start = line.find("{")
        if start < 0:
            continue
        end = line.rfind("}")
        if end < start:
            continue
        try:
            obj = json.loads(line[start : end + 1])
            if obj.get("kind") == "INGESTION_AUDIT":
                audits.append(obj)
        except json.JSONDecodeError:
            continue
    return audits


def analyze(audits: list[dict]) -> tuple[bool, list[str]]:
    """
    Verifica critérios.
    Returns: (pass, list of failure reasons)
    """
    reasons = []
    last_ws_disc = 0
    for i, a in enumerate(audits):
        sw = a.get("snapshots_window", 0)
        ew = a.get("engine_events_window", 0)
        dr = a.get("discard_rate_window", 0)
        silence = a.get("silence_seconds", 0)
        last_ws_disc = a.get("ws_disconnects_total", 0)

        if sw == 0:
            reasons.append(f"amostra {i+1}: snapshots_window=0")
        if ew == 0:
            reasons.append(f"amostra {i+1}: engine_events_window=0")
        if dr >= 0.02:
            reasons.append(f"amostra {i+1}: discard_rate_window={dr} >= 0.02")
        if silence >= 30:
            reasons.append(f"amostra {i+1}: silence_seconds={silence} >= 30")

    if last_ws_disc > 1:
        reasons.append(f"ws_disconnects_total={last_ws_disc} > 1")

    return len(reasons) == 0, reasons


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python scripts/analyze_ingestion_log.py <arquivo.log>", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Arquivo não encontrado: {path}", file=sys.stderr)
        sys.exit(1)

    audits = extract_audit_lines(path)
    if not audits:
        print("INGESTION_STATUS: FAIL", file=sys.stdout)
        print("Motivos:", file=sys.stdout)
        print("  Nenhuma linha INGESTION_AUDIT encontrada no log.", file=sys.stdout)
        sys.exit(1)

    passed, reasons = analyze(audits)

    if passed:
        print("INGESTION_STATUS: PASS", file=sys.stdout)
        sys.exit(0)
    else:
        print("INGESTION_STATUS: FAIL", file=sys.stdout)
        print("Motivos:", file=sys.stdout)
        for r in reasons:
            print(f"  {r}", file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
