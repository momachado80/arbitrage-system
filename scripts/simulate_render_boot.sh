#!/bin/bash
# Simula ambiente Render: STATE_DIR e DATA_DIR em /var/data (não gravável no free tier).
# O sistema deve fazer fallback para /tmp e bootar com sucesso.
#
# Uso: ./scripts/simulate_render_boot.sh

set -e
cd "$(dirname "$0")/.."

export DATA_DIR=/var/data
export STATE_DIR=/var/data/state
export POLYMARKET_TOKENS=0x0000000000000000000000000000000000000001

echo "=== Simulando Render (DATA_DIR=$DATA_DIR, STATE_DIR=$STATE_DIR) ==="
echo "Esperado: fallback para /tmp, create_system boota OK"
echo ""

python -c "
import os
os.environ['DATA_DIR'] = '/var/data'
os.environ['STATE_DIR'] = '/var/data/state'
os.environ['POLYMARKET_TOKENS'] = '0x123'
from src.orchestrator import create_system
sys = create_system(initial_capital=1000.0, token_ids=['0x123'])
rsd = sys.get('resolved_state_dir', '')
rdd = sys.get('resolved_data_dir', '')
boot_err = sys.get('boot_error')
assert '/tmp' in rsd, 'resolved_state_dir deve usar /tmp'
assert boot_err is None, f'boot_error deve ser None: {boot_err}'
print('PASS: create_system bootou com fallback para /tmp')
print('  resolved_state_dir:', rsd)
print('  resolved_data_dir:', rdd)
"

echo ""
echo "=== Simulação Render OK ==="
