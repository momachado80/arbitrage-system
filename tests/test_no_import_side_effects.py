"""
Guardrail: impede regressão de side effects (mkdir, open) em import time.

Import de src.server e src.orchestrator não deve criar diretórios fora de /tmp
nem em paths absolutos proibidos (/var, /home, /opt, etc).

Executa apenas import — sem startup event. Python 3.9+.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Paths absolutos proibidos no import (Render, etc)
FORBIDDEN_PREFIXES = ("/var/", "/home/", "/opt/", "/usr/local/var/", "/etc/")


def _is_forbidden(path: str) -> bool:
    """True se path é absoluto e começa com prefixo proibido."""
    p = str(path).strip()
    if not os.path.isabs(p):
        return False
    return any(p.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def _patched_makedirs(original, calls_store):
    """Wrap que registra chamadas e bloqueia paths proibidos."""

    def wrapper(path, *args, **kwargs):
        path_str = str(path)
        if _is_forbidden(path_str):
            raise AssertionError(f"BLOCKED: makedirs forbidden path during import: {path_str}")
        calls_store.append(path_str)
        return original(path, *args, **kwargs)

    return wrapper


class TestNoImportSideEffects:
    """
    Testa que import de módulos principais não faz I/O em paths proibidos.
    """

    @pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="src.server requires Python 3.11.x; skip on older Python",
    )
    def test_server_import_no_forbidden_makedirs(self):
        """Import de src.server não deve makedirs em /var, /home, /opt."""
        makedirs_calls = []
        mkdir_calls = []

        original_makedirs = os.makedirs
        original_path_mkdir = Path.mkdir

        def patched_path_mkdir(self, *args, **kwargs):
            path_str = str(self)
            if _is_forbidden(path_str):
                raise AssertionError(
                    f"BLOCKED: Path.mkdir forbidden path during import: {path_str}"
                )
            mkdir_calls.append(path_str)
            return original_path_mkdir(self, *args, **kwargs)

        with patch("os.makedirs", side_effect=_patched_makedirs(original_makedirs, makedirs_calls)):
            with patch.object(Path, "mkdir", patched_path_mkdir):
                import src.server  # noqa: F401

        for p in makedirs_calls + mkdir_calls:
            assert not _is_forbidden(p), f"Import criou path proibido: {p}"

    def test_orchestrator_import_no_forbidden_makedirs(self):
        """Import de src.orchestrator não deve makedirs em /var, /home, /opt."""
        makedirs_calls = []

        original_makedirs = os.makedirs
        with patch("os.makedirs", side_effect=_patched_makedirs(original_makedirs, makedirs_calls)):
            import src.orchestrator  # noqa: F401

        for p in makedirs_calls:
            assert not _is_forbidden(p), f"Import criou path proibido: {p}"
