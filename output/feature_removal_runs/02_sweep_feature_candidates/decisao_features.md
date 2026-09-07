# Decisao apos sweep de remocao de features

## Escopo

Foram testados 22 cenarios em validacao cruzada com XGBoost + SMOTE e 18 cenarios no Monte Carlo v1, considerando:

- remocoes individuais;
- blocos socioeconomicos/sensiveis;
- pares correlacionados pela matriz de Spearman;
- combinacoes mais agressivas.

Os arquivos completos estao nesta pasta:

- `cv_sweep_resumo.md`
- `cv_sweep_resultados.csv`
- `cv_sweep_ranking_mesmas_linhas.csv`
- `monte_carlo_v1_resumo.md`
- `monte_carlo_v1_sweep_resultados.csv`
- `monte_carlo_v1_ranking.csv`
- `monte_carlo_v1_ranking_mesmas_linhas.csv`

## Melhor candidato para remover agora

### `CD`

Motivo:

- tem redundancia com `ODD` pela matriz de Spearman;
- `ODD` conversa mais com GAD e parece carregar melhor o sinal clinico;
- melhorou a validacao cruzada controlada;
- melhorou o Monte Carlo v1.

Resultados principais:

- CV mesmas linhas: Kappa `0.3249` vs baseline `0.3042` (`+0.0207`).
- CV mesmas linhas: sensibilidade `41.00` vs baseline `36.00` (`+5.00`).
- Monte Carlo v1 mesmas linhas: Kappa `0.8410` vs baseline `0.8130` (`+0.0281`).
- Monte Carlo v1 mesmas linhas: F1 `87.09` vs baseline `85.67` (`+1.42`).

Leitura: e a remocao mais defensavel porque melhora nos dois mundos: CV e Monte Carlo v1.

## Candidato forte, mas precisa cautela

### `Family History - Psychiatric Diagnosis`

Motivo:

- correlacao fraca com GAD;
- baixa importancia no XGBoost;
- melhorou pouco na CV;
- melhorou muito no Monte Carlo v1.

Resultados:

- CV mesmas linhas: Kappa `0.3145` vs baseline `0.3042` (`+0.0103`).
- Monte Carlo v1 mesmas linhas: Kappa `0.8982` vs baseline `0.8130` (`+0.0853`).

Leitura: e uma candidata muito boa para a proxima rodada isolada, mas o salto no Monte Carlo v1 e grande o suficiente para merecer confirmacao com outro seed/split antes de virar regra final.

## Candidatos contraditorios

### `Poverty Status`

Motivo:

- estatisticamente parecia descartavel pela baixa correlacao com GAD;
- no Monte Carlo v1 controlado melhorou;
- mas na CV controlada piorou bastante e, com todas as amostras validas, tambem ficou pior.

Leitura: nao remover isoladamente ainda. O efeito depende muito do protocolo de avaliacao.

### `Number of Bio. Parents`

Motivo:

- VIF moderado e redundancia com `Race`/`Poverty Status`;
- CV melhora levemente;
- Monte Carlo v1 piora levemente.

Leitura: deixar em observacao, sem remover agora.

### `Number of Siblings`

Motivo:

- CV melhora levemente;
- Monte Carlo v1 piora levemente.

Leitura: efeito pequeno e contraditorio; nao e prioridade.

## Remocoes que parecem arriscadas agora

### `Age`

Remover `Age` piorou CV e Monte Carlo v1.

- CV Kappa: `0.2004` vs `0.3042`.
- Monte Carlo v1 Kappa: `0.6634` vs `0.8130`.

Leitura: manter.

### `Number of Sleep Disturbances`

Apesar do VIF moderado, remover piorou o Monte Carlo v1.

- CV ficou praticamente empatado.
- Monte Carlo v1 Kappa caiu para `0.7274`.

Leitura: manter.

### `Frequency Temper Tantrums`

Mesmo sendo correlacionada com `Frequency Irritable Mood`, remover piorou CV e Monte Carlo v1.

Leitura: manter por enquanto.

### `Frequency Irritable Mood`

CV melhorou ao remover, mas Monte Carlo v1 piorou.

Leitura: manter por enquanto, porque parece importante para estabilidade nos hard samples.

### `ODD`

Monte Carlo v1 melhora ao remover, mas CV nao melhora e `ODD` tem forte relacao com GAD.

Leitura: nao remover agora; se o par `CD`/`ODD` for simplificado, a melhor opcao e remover `CD` e manter `ODD`.

### `Social Phobia`

CV melhora bastante ao remover e Monte Carlo v1 melhora pouco, mas a feature tem forte associacao com GAD e faz sentido clinico.

Leitura: nao remover direto. Vale testar numa rodada propria com justificativa clinica antes de decidir.

### `Number of Sensory Sensitivities`

CV e Monte Carlo v1 melhoram ao remover, mas existe sinal clinico e correlacao com `Social Phobia`.

Leitura: candidata para uma rodada futura, nao para a proxima remocao.

## Blocos testados

### Quatro features fracas/redundantes

Removidas:

- `Poverty Status`
- `Number of Siblings`
- `Family History - Psychiatric Diagnosis`
- `Number of Bio. Parents`

Resultado:

- CV mesmas linhas melhora: Kappa `0.3580` vs `0.3042`.
- Monte Carlo v1 mesmas linhas piora: Kappa `0.7857` vs `0.8130`.
- Monte Carlo v1 com todas as amostras validas piora mais: Kappa `0.6622`.

Leitura: bom para CV, ruim para Monte Carlo v1. Como o v1 ainda esta em uso, essa lista nao deve ser tratada como final sem novo teste controlado.

### Quatro features + `CD`

Resultado:

- melhor cenario na CV: Kappa `0.3790`;
- piora no Monte Carlo v1: Kappa `0.7532` nas mesmas linhas.

Leitura: bom para CV, ruim para v1. Nao usar como limpeza final agora.

### Seis features: quatro + `CD` + `Frequency Temper Tantrums`

Resultado:

- ruim na CV;
- quase empata no Monte Carlo v1 com amostras validas, mas piora nas mesmas linhas.

Leitura: agressivo demais para agora.

## Recomendacao de proxima rodada

Rodada 03 sugerida:

- remover apenas `CD`;
- gerar snapshot `03_sem_cd`;
- rodar CV, Monte Carlo v1 e hard samples;
- comparar contra baseline e contra `01_sem_4_features`.

Rodada 04 sugerida:

- remover `CD` + `Family History - Psychiatric Diagnosis`;
- fazer somente se a rodada `03_sem_cd` confirmar ganho.

Conclusao atual: se o criterio principal for manter estabilidade no Monte Carlo v1, a melhor proxima remocao e `CD`, nao o bloco das 4 features.
