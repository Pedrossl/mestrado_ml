# Rodada 01 - sem 4 features

Features removidas:

- `Poverty Status`
- `Number of Siblings`
- `Family History - Psychiatric Diagnosis`
- `Number of Bio. Parents`

Arquivos desta rodada:

- `feature_ablation_gad.txt`: comparativo XGBoost + SMOTE entre baseline e limpeza.
- `feature_ablation_gad.csv`: mesma comparacao em formato tabular.
- `monte_carlo_v1_pre_feature_cleanup.txt`: resultado v1 antes desta limpeza.
- `monte_carlo_v1_post_feature_cleanup.txt`: resultado v1 depois desta limpeza, ja com comparacao contra o resultado anterior.
- `monte_carlo_v1_comparativo_pre_pos.txt`: leitura curta do antes/depois no v1.
- `hard_samples_v1_post_feature_cleanup.csv`: hard samples gerados pelo v1 apos a limpeza.

Leitura rapida:

- Na validacao cruzada, a limpeza melhora o resultado quando comparada nas mesmas 287 linhas do baseline.
- No Monte Carlo v1, as metricas caem em relacao ao relatorio antigo, mas a comparacao muda tambem o split e os hard samples porque o conjunto limpo passa a ter 306 amostras validas.
