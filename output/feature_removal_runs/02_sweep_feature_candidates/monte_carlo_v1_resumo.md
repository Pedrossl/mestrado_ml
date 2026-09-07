# Monte Carlo v1 por cenario de remocao

Cada cenario refaz split 80/20, seleciona 20 hard samples e roda 200 simulacoes.

## Ranking com amostras validas por cenario

| Cenario | Drops | Amostras | Hard + | Acc sem | Sens sem | F1 sem | Kappa sem | Delta Kappa | Racional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sem_family_history_psych | 1 | 287 | 4 | 96.70 | 93.00 | 91.85 | 0.8982 | +0.0853 | Correlacao fraca com GAD e baixa importancia. |
| sem_odd | 1 | 287 | 5 | 95.15 | 82.80 | 88.85 | 0.8588 | +0.0459 | Alternativa inversa do par CD/ODD para validar qual carrega mais sinal. |
| sem_sensory | 1 | 287 | 5 | 94.62 | 87.20 | 88.70 | 0.8524 | +0.0395 | Redundancia moderada com Social Phobia; testar perda de sinal. |
| sem_cd | 1 | 287 | 4 | 95.03 | 86.38 | 87.09 | 0.8410 | +0.0281 | CD e redundante com ODD; ODD tem maior sinal com GAD. |
| baseline | 0 | 287 | 5 | 93.20 | 84.30 | 85.67 | 0.8130 | +0.0000 | Todas as features atuais; referencia para comparacao. |
| sem_4_cd_tantrums | 6 | 306 | 6 | 92.28 | 83.25 | 86.22 | 0.8094 | -0.0035 | Limpeza intermediaria: baixa relevancia + dois blocos correlacionados. |
| sem_number_siblings | 1 | 287 | 5 | 92.50 | 87.00 | 85.28 | 0.8032 | -0.0097 | Correlacao quase nula com GAD e baixa importancia. |
| sem_bio_parents | 1 | 287 | 5 | 93.05 | 79.40 | 84.46 | 0.8015 | -0.0114 | Correlacao fraca com GAD, VIF moderado e redundancia com Race/Poverty. |
| sem_4_mais_tantrums | 5 | 306 | 6 | 91.50 | 81.92 | 84.78 | 0.7899 | -0.0231 | Primeira limpeza + remocao de tantrums por redundancia com irritabilidade. |
| sem_irritable_mood | 1 | 287 | 5 | 92.20 | 80.90 | 83.36 | 0.7840 | -0.0290 | Alternativa inversa do par irritabilidade/tantrums. |
| sem_tantrums | 1 | 287 | 4 | 93.03 | 82.12 | 82.24 | 0.7801 | -0.0329 | Redundante com Frequency Irritable Mood e ODD. |
| sem_4_mais_cd | 5 | 306 | 6 | 89.80 | 84.83 | 83.27 | 0.7602 | -0.0528 | Primeira limpeza + remocao do par redundante CD/ODD. |
| sem_social_phobia | 1 | 288 | 4 | 92.10 | 83.88 | 80.81 | 0.7596 | -0.0534 | Alternativa do par Social Phobia/Sensory Sensitivities. |
| sem_poverty_status | 1 | 306 | 4 | 91.35 | 85.38 | 79.72 | 0.7440 | -0.0689 | Menor correlacao com GAD e redundancia socioeconomica. |
| sem_sleep | 1 | 287 | 5 | 89.90 | 79.30 | 79.27 | 0.7274 | -0.0856 | VIF moderado; testar se multicolinearidade pesa mais que sinal clinico. |
| sem_age | 1 | 287 | 3 | 90.85 | 78.33 | 71.49 | 0.6634 | -0.1495 | Maior VIF; testar custo de remover feature importante. |
| sem_4_fracas_redundantes | 4 | 306 | 4 | 88.42 | 81.12 | 73.35 | 0.6622 | -0.1507 | Primeira limpeza segura: baixa relacao com GAD e/ou redundancia. |
| sem_features_sensiveis | 4 | 306 | 5 | 85.08 | 85.70 | 74.17 | 0.6409 | -0.1721 | Teste de fairness: remove variaveis sensiveis ou proxy demografico. |

## Ranking controlado pelas mesmas linhas do baseline

| Cenario | Drops | Amostras | Hard + | Acc sem | Sens sem | F1 sem | Kappa sem | Delta Kappa | Racional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sem_family_history_psych | 1 | 287 | 4 | 96.70 | 93.00 | 91.85 | 0.8982 | +0.0853 | Correlacao fraca com GAD e baixa importancia. |
| sem_odd | 1 | 287 | 5 | 95.15 | 82.80 | 88.85 | 0.8588 | +0.0459 | Alternativa inversa do par CD/ODD para validar qual carrega mais sinal. |
| sem_poverty_status | 1 | 287 | 4 | 95.80 | 82.50 | 87.87 | 0.8547 | +0.0417 | Menor correlacao com GAD e redundancia socioeconomica. |
| sem_sensory | 1 | 287 | 5 | 94.62 | 87.20 | 88.70 | 0.8524 | +0.0395 | Redundancia moderada com Social Phobia; testar perda de sinal. |
| sem_cd | 1 | 287 | 4 | 95.03 | 86.38 | 87.09 | 0.8410 | +0.0281 | CD e redundante com ODD; ODD tem maior sinal com GAD. |
| sem_social_phobia | 1 | 287 | 4 | 94.35 | 84.50 | 85.38 | 0.8196 | +0.0067 | Alternativa do par Social Phobia/Sensory Sensitivities. |
| sem_number_siblings | 1 | 287 | 5 | 92.50 | 87.00 | 85.28 | 0.8032 | -0.0097 | Correlacao quase nula com GAD e baixa importancia. |
| sem_bio_parents | 1 | 287 | 5 | 93.05 | 79.40 | 84.46 | 0.8015 | -0.0114 | Correlacao fraca com GAD, VIF moderado e redundancia com Race/Poverty. |
| sem_4_fracas_redundantes | 4 | 287 | 4 | 92.97 | 84.88 | 82.86 | 0.7857 | -0.0273 | Primeira limpeza segura: baixa relacao com GAD e/ou redundancia. |
| sem_irritable_mood | 1 | 287 | 5 | 92.20 | 80.90 | 83.36 | 0.7840 | -0.0290 | Alternativa inversa do par irritabilidade/tantrums. |
| sem_tantrums | 1 | 287 | 4 | 93.03 | 82.12 | 82.24 | 0.7801 | -0.0329 | Redundante com Frequency Irritable Mood e ODD. |
| sem_4_mais_tantrums | 5 | 287 | 3 | 93.83 | 88.50 | 81.35 | 0.7781 | -0.0348 | Primeira limpeza + remocao de tantrums por redundancia com irritabilidade. |
| sem_features_sensiveis | 4 | 287 | 5 | 90.55 | 89.00 | 82.56 | 0.7621 | -0.0509 | Teste de fairness: remove variaveis sensiveis ou proxy demografico. |
| sem_4_cd_tantrums | 6 | 287 | 4 | 92.17 | 84.62 | 81.00 | 0.7618 | -0.0512 | Limpeza intermediaria: baixa relevancia + dois blocos correlacionados. |
| sem_4_mais_cd | 5 | 287 | 5 | 91.08 | 79.10 | 80.98 | 0.7532 | -0.0597 | Primeira limpeza + remocao do par redundante CD/ODD. |
| sem_sleep | 1 | 287 | 5 | 89.90 | 79.30 | 79.27 | 0.7274 | -0.0856 | VIF moderado; testar se multicolinearidade pesa mais que sinal clinico. |
| sem_age | 1 | 287 | 3 | 90.85 | 78.33 | 71.49 | 0.6634 | -0.1495 | Maior VIF; testar custo de remover feature importante. |
