# =============================================================================
# Executa os 3 passos do experimento em sequência
# =============================================================================

import runpy

print("\n" + "=" * 60)
print("  PASSO 1 — Split 80/20")
print("=" * 60)
runpy.run_module('scripts.experimento_hard_samples.passo1_split_80_20', run_name='__main__')

print("\n" + "=" * 60)
print("  PASSO 2 — Identificar Hard Samples")
print("=" * 60)
runpy.run_module('scripts.experimento_hard_samples.passo2_hard_samples', run_name='__main__')

print("\n" + "=" * 60)
print("  PASSO 3 — Monte Carlo com e sem Smoothing")
print("=" * 60)
runpy.run_module('scripts.experimento_hard_samples.passo3_monte_carlo', run_name='__main__')
