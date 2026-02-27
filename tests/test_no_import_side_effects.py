"""
Guardrail: impede regressão de side effects (mkdir, open) em import time.

Import de src.server e src.orchestrator não deve criar diretórios fora de /tmp
nem em paths absolutos proibidos. Permite apenas /tmp e relativos.

Executa apenas import — sem startup event. Python 3.9+.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _is_allowed(path: str) -> bool:
    """
    True se path é permitido: relativo ou absoluto sob /tmp.
    Bloqueia /var, /home, /opt e qualquer absoluto fora de /tmp.
    """
    p = os.path.normpath(str(path).strip())
    if not os.path.isabs(p):
        return True
    return p.startswith("/tmp/") or p == "/tmp"


def _is_forbidden(path: str) -> bool:
    """True se path é absoluto fora de /tmp (equivale a não permitido)."""
    return not _is_allowed(path)


def _patched_makedirs(original, calls_store):
    """Wrap que registra chamadas e bloqueia paths fora de /tmp e relativos."""

    def wrapper(path, *args, **kwargs):
        path_str = str(path)
        if _is_forbidden(path_str):
            raise AssertionError(
                f"BLOCKED: makedirs forbidden path during import: {path_str!r}"
            )
        calls_store.append(path_str)
        return original(path, *args, **kwargs)

    return wrapper




class TestNoImportSideEffects:
    """
    Testa que import de módulos principais não faz I/O em paths proibidos.
    Roda em Python 3.9+.
    """

    def test_server_import_no_forbidden_makedirs(self):
        """Import de src.server não deve makedirs fora de /tmp nem em absolutos proibidos."""
        makedirs_calls = []
        mkdir_calls = []

        original_makedirs = os.makedirs
        original_path_mkdir = Path.mkdir

        def patched_path_mkdir(self, *args, **kwargs):
            path_str = str(self)
            if _is_forbidden(path_str):
                raise AssertionError(
                    f"BLOCKED: Path.mkdir forbidden path during import: {path_str!r}"
                )
            mkdir_calls.append(path_str)
            return original_path_mkdir(self, *args, **kwargs)

        with patch.dict(os.environ, {"SKIP_PYTHON_VERSION_CHECK": "1"}, clear=False):
            with patch("os.makedirs", side_effect=_patched_makedirs(original_makedirs, makedirs_calls)):
                with patch.object(Path, "mkdir", patched_path_mkdir):
                    import src.server  # noqa: F401

        for p in makedirs_calls + mkdir_calls:
            assert _is_allowed(p), f"Import criou path proibido: {p!r}"

    def test_orchestrator_import_no_forbidden_makedirs(self):
        """Import de src.orchestrator não deve makedirs fora de /tmp."""
        makedirs_calls = []

        original_makedirs = os.makedirs
        with patch("os.makedirs", side_effect=_patched_makedirs(original_makedirs, makedirs_calls)):
            import src.orchestrator  # noqa: F401

        for p in makedirs_calls:
            assert _is_allowed(p), f"Import criou path proibido: {p!r}"
