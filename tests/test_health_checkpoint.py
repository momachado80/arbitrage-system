"""
Guardrail: health checkpoint path deve permanecer sob /tmp.
Testa a mesma lógica usada em src.server._checkpoint_path_safe_under_tmp.
"""

import os


def _checkpoint_path_safe_under_tmp(path: str) -> bool:
    """Cópia da lógica em server para teste isolado."""
    try:
        real = os.path.realpath(path)
        return real.startswith("/tmp") or real.startswith("/private/tmp")
    except Exception:
        return False


def test_checkpoint_path_stays_under_tmp():
    """Garante que path de state dir fica sob /tmp."""
    state_dir = "/tmp/polymarket_state"
    path = os.path.join(state_dir, "health_checkpoint.json")
    assert path.endswith("health_checkpoint.json")
    assert _checkpoint_path_safe_under_tmp(path), f"path {path} fora de /tmp"


def test_checkpoint_path_rejects_outside_tmp():
    """Garante que paths fora de /tmp são rejeitados."""
    assert not _checkpoint_path_safe_under_tmp("/etc/health_checkpoint.json")
    assert not _checkpoint_path_safe_under_tmp("/home/foo/health_checkpoint.json")
    assert not _checkpoint_path_safe_under_tmp("/var/polymarket_state/health_checkpoint.json")
