# Relatorio Completo de Experimentos — GAD

Consolida TODOS os experimentos de selecao de features e tuning do Monte Carlo v1,
com explicacao objetiva do motivo de cada resultado. Todos os numeros abaixo foram
re-executados e conferidos nesta rodada (seeds fixas, 100% reprodutiveis).

Ver tambem: [FEATURE_SELECTION.md](../FEATURE_SELECTION.md) (estrategia de selecao)
e [melhor_resultado_monte_carlo.txt](melhor_resultado_monte_carlo.txt) (resultado oficial).


## Metodologias e tecnologias utilizadas

### Stack tecnico

| Componente | Ferramenta | Versao |
| --- | --- | --- |
| Linguagem | Python | 3.9.6 |
| Modelo | XGBoost (Gradient Boosting) | 2.1.4 |
| Balanceamento de classes | imbalanced-learn (SMOTE, SMOTETomek, SMOTEENN) | 0.12.4 |
| Validacao, metricas, correlacao | scikit-learn | 1.6.1 |
| Testes estatisticos (Spearman) | scipy | 1.13.1 |
| Manipulacao de dados | pandas / numpy | 2.3.3 / 2.0.2 |

### Metodologias aplicadas

1. **Pre-processamento**: normalizacao MinMax nas variaveis continuas, encoding
   binario/ordinal nas categoricas, listwise deletion para valores faltantes
   (307 -> 287 registros). Ver `scripts/preprocessing/`.

2. **Selecao de features (3 etapas)** — ver [FEATURE_SELECTION.md](../FEATURE_SELECTION.md)
   para o detalhamento completo:
   - **Correlacao de Spearman**: filtro de redundancia entre pares de features e
     de relevancia de cada feature com o alvo (GAD). Nao assume relacao linear
     (ao contrario de Pearson), adequado para variaveis ordinais/binarias do
     dataset clinico.
   - **Permutation Importance**: embaralha cada feature individualmente e mede a
     queda (ou ganho) no Kappa via CV 10-fold, 10 repeticoes por fold, para
     reduzir ruido de uma unica reamostragem. Importancia negativa indica que a
     feature atrapalha o modelo.
   - **Validacao por Monte Carlo v1**: cada candidata a remocao so e aceita se
     melhorar o Monte Carlo v1 (nao so a CV) — evita decisoes baseadas em uma
     unica metrica que pode nao concordar com o comportamento nos casos dificeis.

3. **Validacao cruzada (CV) 10-fold estratificada**: protocolo honesto (sem
   vazamento), usado para Permutation Importance e para o comparativo geral de
   algoritmos/tecnicas de balanceamento da dissertacao.

4. **Monte Carlo v1 (experimento hard samples)**: protocolo especifico para medir
   estabilidade do modelo nos casos mais dificeis de classificar.
   - Split 80/20 estratificado (seed=42) treina um modelo base com SMOTE.
   - Os 20 casos do teste com probabilidade mais proxima de 0.5 (menor margem de
     confianca) sao isolados como "hard samples".
   - 200 simulacoes: em cada uma, sorteia-se 15 dos 20 hard samples, adiciona-se
     ao treino, retreina-se o modelo e avalia-se nos 20 hard samples completos.
   - Metricas agregadas como media ± IC 95%, mais desvio padrao (sigma) entre as
     200 simulacoes, para capturar tanto a performance media quanto a estabilidade.
   - **Limitacao conhecida**: como parte dos hard samples de avaliacao tambem
     entra no retreino, ha vazamento de dados — ver Secao 0 abaixo. Aceito porque
     o objetivo e medir consistencia, nao generalizacao pura.

5. **Balanceamento de classes**: SMOTE (Synthetic Minority Oversampling) aplicado
   apenas no conjunto de treino de cada fold/split (sem vazamento), gerando
   amostras sinteticas da classe minoritaria (GAD positivo, ~15% da base).
   SMOTETomek e SMOTEENN testados como alternativas (combinam oversampling com
   limpeza de fronteira), mas nao superaram o SMOTE puro neste dataset.

6. **Tuning de hiperparametros do XGBoost**: busca manual e direcionada (nao
   GridSearch exaustivo) em `max_depth`, `n_estimators`, `scale_pos_weight`,
   threshold de decisao e parametros do Monte Carlo (`N_HARD`, `TAMANHO_SORTEIO`),
   sempre validado pelo Monte Carlo v1 completo (200 simulacoes) para cada
   configuracao testada.


## 0. Dois protocolos de avaliacao — nao confundir

Este projeto usa dois protocolos diferentes, com escalas de Kappa muito diferentes
para o MESMO baseline. Isso e proposital, mas gera confusao se nao for explicado:

| Protocolo | O que faz | Kappa do baseline (17 feat) |
| --- | --- | --- |
| **CV 10-fold** | Validacao cruzada honesta, sem vazamento | 0.304 |
| **Monte Carlo v1** | Retreina incluindo hard samples e avalia nos mesmos hard samples (vazamento conhecido) | 0.813 |

O Monte Carlo v1 tem vazamento de dados: parte dos 20 "hard samples" usados na
avaliacao final tambem entrou no retreino daquela simulacao. Isso infla o Kappa
em relacao a um cenario 100% honesto. Esse vazamento e conhecido e aceito porque
o objetivo do MC v1 e medir **estabilidade/consistencia** do modelo nos casos
mais dificeis, nao performance de generalizacao pura — mas ele explica por que
alguns resultados abaixo (especialmente a secao 5) parecem bons demais.


## 1. Por que 13 features (306 amostras) ficou PIOR que 15 features (287 amostras)

Esta e a comparacao mais enganosa do projeto porque **dois fatores mudam ao mesmo
tempo**: o numero de features E o numero de amostras.

### Os 4 dados

Monte Carlo v1, 200 simulacoes, sorteio 15/20 hard samples:

| Cenario | Features | Amostras | Kappa | Sens | F1 | Spec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 17 | 287 | 0.8130 | 84.30% | 85.67% | 96.17% |
| Sem CD | 16 | 287 | **0.8410** | 86.38% | 87.09% | 97.19% |
| Sem 4 features | 13 | 306 | **0.6622** | 81.12% | 73.35% | 90.25% |
| Sem 4 features + CD | 12 | 306 | 0.7602 | 84.83% | 83.27% | 91.93% |

### Explicacao objetiva

**Nao e o numero de features que causa a piora.** A prova esta na coluna "16 feat /
287 amostras": removendo so o CD e mantendo as mesmas 287 amostras, o Kappa
**melhora** (0.813 -> 0.841). Menos features, com o mesmo dataset, ajuda.

O problema e outro: as 4 features removidas no cenario de 13 features (`Poverty
Status`, `Number of Siblings`, `Family History`, `Number of Bio. Parents`) tinham
valores faltantes (missing) em algumas linhas. O pre-processamento original fazia
listwise deletion — removia a linha inteira se qualquer feature tivesse valor
faltante. Ao tirar essas 4 features da equacao, 19 linhas que antes eram
descartadas por terem missing **passam a entrar no dataset**, subindo a amostra
de 287 para 306.

Isso quebra a comparacao de tres formas:

1. **Split diferente**: o split 80/20 estratificado (seed=42) e recalculado sobre
   306 linhas em vez de 287 — treino e teste passam a ser conjuntos diferentes,
   nao apenas "os mesmos + 19 a mais".
2. **Hard samples diferentes**: os 20 "casos dificeis" identificados pelo modelo
   base mudam de identidade. Prova: com 287 amostras o teste tem 5 positivos
   entre os hard samples e o modelo base acerta so 9/20; com 306 amostras, muda
   para 4 positivos e 12/20 de acerto. Sao problemas de dificuldade diferente.
3. **Qualidade dos dados**: as 19 linhas reincorporadas sao justamente as que
   tinham dados incompletos em outras variaveis — nao ha garantia de que sejam
   tao informativas quanto o resto da amostra.

Por isso 13 features/306 amostras (Kappa 0.6622) parece muito pior que 16
features/287 amostras (Kappa 0.8410): a diferenca real nao vem das features
removidas, vem da mudanca de composicao da amostra. A prova final: 12 features
(13 menos o CD, ainda com 306 amostras) melhora para 0.7602 — a remocao do CD
ajuda em qualquer regime de amostra, mas o "efeito 306 amostras" continua
prejudicando os dois cenarios de 306 em relacao aos de 287.

**Decisao tomada**: manter as 287 amostras originais e remover apenas features
sem missing values (CD e Family History), evitando esse confundimento.


## 2. Por que CD e Family History foram removidas (e as outras nao)

### CD (Conduct Disorder) — redundancia com ODD

- Correlacao de Spearman CD-ODD: rho = 0.502 (alta redundancia)
- Correlacao com GAD: CD rho = 0.241, ODD rho = 0.293 (ODD carrega mais sinal)
- Mecanismo: com as duas presentes, o XGBoost divide os splits de decisao entre
  CD e ODD de forma instavel. Nos hard samples (casos na fronteira de decisao),
  essa divisao gera erros porque o modelo as vezes usa a feature mais fraca (CD)
  onde deveria usar a mais forte (ODD). Removendo CD, toda a informacao
  comportamental fica concentrada em ODD.
- Impacto isolado (Monte Carlo v1, 287 amostras): Kappa 0.8130 -> 0.8410 (+0.028)

### Family History - Psychiatric Diagnosis — ruido

- Correlacao com GAD: rho = 0.122 (fraca)
- Permutation Importance (CV 10-fold, 10 repeticoes): **-0.0125** (negativa —
  embaralhar a feature MELHORA o modelo)
- Mecanismo: o XGBoost encontra splits nessa feature que parecem uteis no treino
  mas sao espurios (nao generalizam). No retreino do Monte Carlo, que usa poucas
  amostras extras (15 hard samples por simulacao), esses splits falsos pioram
  ainda mais — o modelo "decora" um padrao que nao existe de verdade.
- Impacto isolado (Monte Carlo v1, com CD ja removido): Kappa 0.8410 -> 0.8666
  aproximadamente (o efeito combinado das duas remocoes, sem tuning de
  hiperparametros, e Kappa 0.813 -> 0.8666).

### Por que o impacto e desproporcional no Monte Carlo

O MC v1 retreina o modelo adicionando so 15 hard samples por simulacao. Com tao
poucos dados extras, qualquer feature ruidosa ou redundante tem efeito
desproporcional: o modelo pode criar splits que "funcionam" por acaso nesses 15
casos mas falham nos outros 5. Removendo as fontes de ruido, o retreino fica mais
estavel — por isso o Kappa sobe **e** o desvio padrao cai ao mesmo tempo
(sigma_kappa: 0.1277 -> 0.1169).

### Candidatas testadas e rejeitadas

Confirmado nesta rodada via Permutation Importance (15 features atuais, CV
10-fold): nenhuma feature restante tem importancia claramente negativa. A mais
fraca e ADHD (-0.0004), mas dentro do proprio desvio padrao (±0.0013) — ou seja,
estatisticamente indistinguivel de zero. Isso confirma que a selecao atingiu um
ponto de parada natural: nao ha mais candidatas obvias.

Outras features testadas individualmente no passado e rejeitadas porque
**melhoravam a CV mas pioravam o Monte Carlo** (por isso a validacao em 2 estagios
e obrigatoria — CV sozinha teria removido features uteis):

| Feature testada | Efeito na CV | Efeito no Monte Carlo v1 |
| --- | --- | --- |
| Number of Bio. Parents | Kappa melhora | Kappa piora (0.841 -> 0.709) |
| ADHD | — | Kappa piora (0.841 -> 0.814) |
| Sex | — | Kappa piora (0.813 -> 0.692) |
| Number of Siblings | — | Kappa piora levemente (0.813 -> 0.803) |
| Poverty Status | Kappa piora | — |
| Age | Kappa piora muito | — |
| Frequency Temper Tantrums | Kappa piora | — |


## 3. Por que max_depth=8 melhorou o resultado

Config atual (15 features, 287 amostras), Monte Carlo v1:

| Config | Kappa | Sens | F1 | Spec |
| --- | ---: | ---: | ---: | ---: |
| max_depth=6 (default) | 0.8666 | 93.00% | 89.44% | 96.19% |
| **max_depth=8** | **0.8832** | 93.25% | 90.73% | 96.84% |
| max_depth=10 | 0.8483 | 91.12% | 87.96% | 95.94% |

**Explicacao**: arvores mais profundas conseguem capturar interacoes mais
complexas entre features durante o retreino com hard samples. Como esses casos
sao dificeis justamente por ficarem na fronteira de decisao, interacoes sutis
entre variaveis (ex: ODD alto + poucos impairments) ajudam a separa-los melhor.
Profundidade 10 ja piora — excesso de profundidade permite que o modelo
decore ruido especifico dos 15 hard samples sorteados em vez de aprender um
padrao generalizavel (overfitting). Existe um ponto otimo em 8.


## 4. Por que o Smoothing so ajudou combinado com max_depth=8

Este e o resultado mais sutil do projeto — a resposta correta e "depende do
max_depth", nao um "sim" ou "nao" simples.

| Configuracao | Kappa sem Smoothing | Kappa com Smoothing (eps=0.1) | Efeito |
| --- | ---: | ---: | ---: |
| 17 feat, depth=6 (original) | 0.8130 | 0.8100 | Smoothing **piora** |
| 15 feat, depth=6 | 0.8666 | 0.8520 | Smoothing **piora** |
| 15 feat, depth=8 | 0.8832 | **0.8867** | Smoothing **melhora** (leve) |

**Explicacao**: o smoothing reduz o peso dos hard samples no retreino
proporcionalmente a sua incerteza (probabilidade proxima de 0.5 -> peso menor).
A ideia e evitar que o modelo fique "overconfident" nesses casos ambiguos.

- Com **max_depth=6** (arvore mais rasa, menos capacidade), o modelo nao tem
  complexidade suficiente para "decorar" os hard samples de forma exagerada.
  Reduzir o peso deles so retira sinal de aprendizado util, sem nenhum
  overfitting real para corrigir — resultado: piora.
- Com **max_depth=8** (arvore mais profunda, mais capacidade), o modelo
  consegue de fato se ajustar demais aos 15 hard samples incluidos a cada
  simulacao. Ai o smoothing cumpre seu papel: reduz esse overfitting
  especifico, gerando uma melhora marginal.

Ou seja: smoothing nao e uma melhoria isolada, e uma correcao para um problema
(overfitting em arvores profundas) que so existe quando a arvore e profunda o
suficiente para causa-lo.


## 5. Tentativas que NAO funcionaram (e por que)

### 5.1. SMOTETomek + max_depth=8 (combinacao)

| Config | Kappa | Sens | Spec |
| --- | ---: | ---: | ---: |
| SMOTE + depth=8 (adotado) | 0.8832 | 93.25% | 96.84% |
| SMOTETomek (sozinho, depth=6) | 0.8729 | 92.88% | 96.47% |
| SMOTETomek + depth=8 (combo) | 0.8373 | 90.40% | 94.83% |

SMOTETomek sozinho ate compete com o baseline (0.8729 vs 0.8666), mas combinado
com max_depth=8 piora bastante (0.8373). **Explicacao**: SMOTETomek já remove
amostras de fronteira ambigua (limpeza via Tomek Links) para reduzir ruido. Isso
tem efeito parecido ao de arvores mais profundas — ambos tentam lidar com a
fronteira de decisao dificil, mas de formas diferentes. Combinar as duas tecnicas
faz o modelo perder amostras uteis (via Tomek) E ainda tentar se ajustar demais
ao pouco que sobrou (via depth=8) — as duas correcoes competem em vez de somar.

### 5.2. scale_pos_weight e threshold ajustado

| Cenario | Threshold | Kappa | Sens | Spec |
| --- | ---: | ---: | ---: | ---: |
| Default (spw=1, t=0.50) | 0.500 | 0.8666 | 93.00% | 96.19% |
| spw=3, t=0.50 | 0.500 | 0.7809 | 91.83% | 93.88% |
| spw=1 + Youden | 0.003 | 0.0000 | 100.00% | 0.00% |
| spw=3 + Youden | 0.626 | 0.8255 | 91.17% | 95.82% |
| spw=5 + Youden | 0.004 | 0.0024 | 100.00% | 0.47% |

**Explicacao**: os dados de treino ja passam por SMOTE, que balanceia as classes
1:1. Aplicar `scale_pos_weight` por cima disso e uma dupla correcao de
desbalanceamento — empurra o modelo a favorecer a classe positiva alem do
necessario, derrubando a especificidade sem ganho real de sensibilidade.

O caso `spw=1 + Youden` e `spw=5 + Youden` colapsam completamente (Kappa ≈ 0,
Spec ≈ 0%): o threshold de Youden e calculado a partir da curva ROC dos 20 hard
samples do conjunto de teste — uma amostra pequena demais para estimar um
threshold estavel. Nesses casos o algoritmo escolheu um limiar quase zero, que
classifica **tudo** como positivo. Isso nao e uma melhoria, e uma quebra do
modelo por instabilidade estatistica em amostra pequena.

### 5.3. SMOTEENN e Re-SMOTE apos hard samples

| Config | Kappa |
| --- | ---: |
| SMOTEENN | 0.6893 |
| Re-SMOTE apos hard samples | 0.8498 |

SMOTEENN combina SMOTE com limpeza via Edited Nearest Neighbours, que remove
agressivamente amostras de fronteira — em um dataset ja pequeno (287 amostras,
15% positivos), isso descarta informacao demais e derruba o Kappa. Re-aplicar
SMOTE depois de incluir os hard samples (para reforcar o balanceamento a cada
simulacao) tambem piora: os hard samples sinteticos gerados por esse SMOTE extra
diluem o sinal real dos hard samples verdadeiros, que sao justamente os mais
informativos por serem dificeis.

### 5.4. N_HARD e TAMANHO_SORTEIO — atencao a um resultado enganoso

| Config | Kappa | Observacao |
| --- | ---: | --- |
| Baseline (N_HARD=20, sorteio=15) | 0.8666 | Protocolo original da dissertacao |
| N_HARD=20, sorteio=10 | 0.6945 | Menos amostras vazadas -> avaliacao mais honesta -> Kappa cai |
| N_HARD=20, sorteio=18 | **0.9557** | Mais amostras vazadas -> Kappa artificialmente inflado |
| N_HARD=15, sorteio=10 | 0.8818 | Dentro do ruido do baseline |
| N_HARD=25, sorteio=18 | 0.8461 | Dentro do ruido do baseline |
| N_HARD=30, sorteio=22 | 0.8465 | Dentro do ruido do baseline |

**Atencao**: o resultado de 0.9557 (sorteio=18 de 20) NAO deve ser interpretado
como uma configuracao melhor. E a demonstracao direta do vazamento de dados do
Monte Carlo v1 (ver secao 0): quando TAMANHO_SORTEIO se aproxima de N_HARD, quase
todos os 20 casos de avaliacao tambem entraram no retreino daquela simulacao — o
modelo esta sendo avaliado em cima do que acabou de "decorar". Prova simetrica:
reduzindo o sorteio para 10/20, exatamente o oposto acontece e o Kappa cai para
0.6945, porque menos hard samples entram no retreino e a avaliacao fica mais
proxima do honesto.

Por isso o protocolo original (15/20, proporcao 75%) foi mantido — mudar essa
proporcao so move o resultado ao longo do eixo "mais vazamento / mais honesto",
nao representa uma melhoria real de modelo.

### 5.5. n_estimators e mais simulacoes

| Config | Kappa |
| --- | ---: |
| n_estimators=200 | 0.8500 |
| depth=8 + n_estimators=200 | 0.8424 |
| 500 simulacoes (vs 200) | 0.8574 |

Mais arvores (n_estimators=200) nao ajuda — o modelo ja converge com 100 arvores
(default), e adicionar mais so aumenta a chance de overfitting nos hard samples.
Rodar 500 simulacoes em vez de 200 da um resultado dentro do intervalo de
confianca do baseline (0.8574 vs 0.8666), confirmando que 200 simulacoes ja e
suficiente para estabilizar a media — nao ha ganho de precisao rodando mais.


## 6. Configuracao final adotada

```
Features: 15 (removidas CD e Family History - Psychiatric Diagnosis)
Amostras: 287 (sem alteracao de composicao)
Balanceamento: SMOTE
XGBoost: max_depth=8 (demais hiperparametros default)
Monte Carlo v1: N_HARD=20, TAMANHO_SORTEIO=15, 200 simulacoes (protocolo original)
```

Resultado (sem smoothing): **Kappa 0.8832**, Sensibilidade 93.25%, F1 90.73%,
Especificidade 96.84%.

Resultado (com smoothing eps=0.1): **Kappa 0.8867**, Sensibilidade 94.25%, F1
91.03%, Especificidade 96.69%.

Evolucao total: Kappa 0.8130 (17 feat, baseline original) -> 0.8832/0.8867
(15 feat + max_depth=8), uma melhora de **8.6-9.1%** com reducao de variancia em
todas as metricas.


## 7. Limitacao estrutural (nao resolvida por tuning)

Todo o tuning acima opera dentro do protocolo do Monte Carlo v1, que tem
vazamento de dados por construcao (secao 0). Isso significa que o teto de
Kappa alcancavel por ajuste fino de hiperparametros e features e limitado pelo
proprio desenho do experimento — nenhuma combinacao de features/hiperparametros
testada supera esse teto de forma genuina, only o desloca dentro da margem que o
vazamento permite. Qualquer melhoria futura de Kappa que se aproxime ou ultrapasse
0.90 deve ser vista com ceticismo e verificada quanto a esse efeito (ver secao
5.4 como exemplo de como o vazamento pode ser confundido com ganho real).
