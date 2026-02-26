"""
Módulo de seed global para reprodutibilidade.
Usado por src.server e scripts/test_seed_determinism.py.
"""

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """
    Configura seed global para reprodutibilidade.
    Chamar antes de qualquer engine ou geração de números aleatórios.

    IMPORTANTE: PYTHONHASHSEED deve ser setado ANTES do processo iniciar
    para que hash() seja determinístico. Dentro deste processo, setar
    os.environ["PYTHONHASHSEED"] só afeta subprocessos.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
