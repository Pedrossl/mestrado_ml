# Rodadas de limpeza de features

Esta pasta guarda snapshots dos resultados a cada rodada de remocao de features.

Convencao:

- `01_sem_4_features/`: primeira rodada, removendo quatro features fracas/redundantes.
- `02_sweep_feature_candidates/`: varredura ampla com remocoes individuais, blocos correlacionados e Monte Carlo v1.
- Proximas rodadas devem usar nomes sequenciais, por exemplo `03_sem_cd/`.

Cada rodada deve conter:

- lista das features removidas;
- comparativo de validacao cruzada;
- comparativo Monte Carlo v1 quando aplicavel;
- hard samples gerados naquela rodada, quando houver.
