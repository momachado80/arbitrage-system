#!/usr/bin/env python3
"""
Teste formal de determinismo do seed global.
Valida que _set_global_seed() torna o ambiente reprodutível.

Uso:
    PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_seed_determinism.py > run1.json
    PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_seed_determinism.py > run2.json
    diff run1.json run2.json   # esperado: nenhuma diferença

    PYTHONHASHSEED=43 RANDOM_SEED=43 python scripts/test_seed_determinism.py > run3.json
    diff run1.json run3.json   # esperado: diferenças detectadas

IMPORTANTE: PYTHONHASHSEED deve ser setado ANTES do processo iniciar
para que hash() seja determinístico.
"""

import json
import os
import sys
from pathlib import Path

# Garantir que o projeto está no path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Verificar PYTHONHASHSEED ANTES de qualquer chamada que possa alterá-lo
PYTHONHASHSEED_SET_BEFORE_LAUNCH = "PYTHONHASHSEED" in os.environ

seed = int(os.environ.get("RANDOM_SEED", "42"))

# Chamar seed ANTES de qualquer geração de números aleatórios
from src.seed import set_global_seed
set_global_seed(seed)

# Gerar valores determinísticos
import random
random_values = [random.random() for _ in range(5)]

numpy_values = None
try:
    import numpy as np
    numpy_values = [float(x) for x in np.random.rand(5)]
except ImportError:
    pass

hash_value = str(hash("arbitrage_test"))

torch_values = None
try:
    import torch
    torch_values = [float(x) for x in torch.rand(5).tolist()]
except ImportError:
    pass

out = {
    "seed": seed,
    "pythonhashseed_set_before_launch": PYTHONHASHSEED_SET_BEFORE_LAUNCH,
    "random": random_values,
    "numpy": numpy_values,
    "hash": hash_value,
    "torch": torch_values,
}
print(json.dumps(out, indent=2))
