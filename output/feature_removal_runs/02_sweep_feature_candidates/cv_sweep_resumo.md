# Sweep de remocao de features

Modelo: XGBoost + SMOTE com 10-fold CV.

## Baseline

- Amostras: 287
- Features: 17
- Sensibilidade: 36.00 +/- 13.99
- Especificidade: 92.18 +/- 4.52
- F1: 39.24 +/- 12.41
- Kappa: 0.3042 +/- 0.1370
- AUC: 0.7609 +/- 0.0996

## Ranking controlado pelas mesmas linhas do baseline

| Cenario | Drops | Sens. | F1 | Kappa | AUC | Delta Kappa | Racional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sem_4_mais_cd | 5 | 41.50 | 45.44 | 0.3790 | 0.7331 | +0.0748 | Primeira limpeza + remocao do par redundante CD/ODD. |
| sem_social_phobia | 1 | 44.00 | 45.94 | 0.3742 | 0.7412 | +0.0700 | Alternativa do par Social Phobia/Sensory Sensitivities. |
| sem_irritable_mood | 1 | 39.50 | 44.77 | 0.3685 | 0.7836 | +0.0643 | Alternativa inversa do par irritabilidade/tantrums. |
| sem_4_fracas_redundantes | 4 | 39.00 | 43.67 | 0.3580 | 0.7395 | +0.0538 | Primeira limpeza segura: baixa relacao com GAD e/ou redundancia. |
| sem_number_siblings | 1 | 36.00 | 41.25 | 0.3329 | 0.7498 | +0.0287 | Correlacao quase nula com GAD e baixa importancia. |
| sem_bio_parents | 1 | 38.50 | 41.99 | 0.3310 | 0.7218 | +0.0269 | Correlacao fraca com GAD, VIF moderado e redundancia com Race/Poverty. |
| sem_sensory | 1 | 38.00 | 41.00 | 0.3264 | 0.7532 | +0.0222 | Redundancia moderada com Social Phobia; testar perda de sinal. |
| sem_cd | 1 | 41.00 | 41.52 | 0.3249 | 0.7493 | +0.0207 | CD e redundante com ODD; ODD tem maior sinal com GAD. |
| sem_4_e_sensiveis | 6 | 41.50 | 40.65 | 0.3145 | 0.7162 | +0.0103 | Primeira limpeza + retirada de variaveis sensiveis remanescentes. |
| sem_family_history_psych | 1 | 38.50 | 40.16 | 0.3145 | 0.7440 | +0.0103 | Correlacao fraca com GAD e baixa importancia. |
| sem_status_race_bio | 3 | 39.00 | 40.80 | 0.3119 | 0.7063 | +0.0078 | Bloco socioeconomico/demografico correlacionado e sensivel. |
| sem_sleep | 1 | 36.00 | 38.69 | 0.3076 | 0.7479 | +0.0034 | VIF moderado; testar se multicolinearidade pesa mais que sinal clinico. |
| sem_odd | 1 | 36.50 | 38.34 | 0.3024 | 0.7504 | -0.0017 | Alternativa inversa do par CD/ODD para validar qual carrega mais sinal. |
| sem_features_sensiveis | 4 | 39.50 | 38.76 | 0.2779 | 0.7080 | -0.0262 | Teste de fairness: remove variaveis sensiveis ou proxy demografico. |
| sem_4_cd_tantrums_sleep | 7 | 32.00 | 35.78 | 0.2564 | 0.7068 | -0.0478 | Limpeza mais agressiva incluindo feature com VIF moderado. |
| sem_4_cd_tantrums | 6 | 32.00 | 34.91 | 0.2527 | 0.7184 | -0.0514 | Limpeza intermediaria: baixa relevancia + dois blocos correlacionados. |
| sem_poverty_status | 1 | 34.00 | 34.21 | 0.2517 | 0.7275 | -0.0525 | Menor correlacao com GAD e redundancia socioeconomica. |
| sem_4_mais_tantrums | 5 | 31.50 | 33.63 | 0.2416 | 0.7064 | -0.0626 | Primeira limpeza + remocao de tantrums por redundancia com irritabilidade. |
| sem_age | 1 | 29.50 | 30.34 | 0.2004 | 0.7171 | -0.1038 | Maior VIF; testar custo de remover feature importante. |
| sem_tantrums | 1 | 27.00 | 27.94 | 0.1911 | 0.7269 | -0.1131 | Redundante com Frequency Irritable Mood e ODD. |
| sem_10_agressivo | 10 | 27.50 | 27.81 | 0.1688 | 0.6774 | -0.1353 | Teste extremo para medir quanto sinal se perde com uma limpeza grande. |
