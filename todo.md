# TODO - Mestrado ML

## Concluido

### Pre-processamento
- [x] Trocar variavel SEX para codificacao binaria (0 e 1)
- [x] Script de normalizacao de dados (`normalizacao.py`)

### Algoritmos Implementados
- [x] **ADTree** - Com 4 tecnicas de balanceamento
- [x] **XGBoost** - Com 4 tecnicas de balanceamento
- [x] **SVM** - Com 4 tecnicas de balanceamento
- [x] **Comparativo dos 3 algoritmos** (`comparativo_algoritmos.py`)

### Tecnicas de Balanceamento (todas implementadas)
- [x] Sem balanceamento (baseline)
- [x] Class Weighting
- [x] SMOTE (corrigido - sem data leakage)
- [x] Undersampling (corrigido - sem data leakage)

### Metricas Implementadas
- [x] Accuracy, Sensitivity, Specificity, PPV, NPV
- [x] F1-Score, Kappa
- [x] Matriz de confusao
- [x] Graficos e tabelas comparativas
- [x] **Intervalos de confianca** (IC 95% com desvio padrao entre folds)
- [x] **Feature Importance** - XGBoost com SMOTE (media dos 10 folds)
- [x] Testes para **GAD** e **SAD**

### Organizacao de Outputs
- [x] Estrutura de pastas separada por GAD/SAD dentro de cada algoritmo
- [x] Scripts configurados para salvar automaticamente na pasta correta (GAD ou SAD)
- [x] Outputs organizados: SVM/GAD, SVM/SAD, XGBoost/GAD, XGBoost/SAD, etc.

---

## 🔴 URGENTE - Pode Derrubar na Banca (Fazer AGORA)

### Validacao e Metricas Essenciais
- [ ] **Curva ROC + AUC** (~2h) - Padrao ouro em classificacao medica. Permite analisar trade-off Sensitivity/Specificity e escolher threshold otimo
- [ ] **Validacao no conjunto de teste holdout** (~1h) - Usar as 307 amostras de teste que ainda nao foram usadas para validar generalizacao
- [ ] **Teste de significancia estatistica** (~1h) - McNemar ou Wilcoxon para provar que XGBoost > SVM nao e so acaso

### Analise de Resultados
- [ ] **Analise detalhada de erros** (~3h) - Identificar caracteristicas dos falsos positivos/negativos. Quais criancas o modelo erra? Por que?
- [ ] **Matriz de confusao normalizada** (~1h) - Visualizar proporcoes de acertos/erros de forma mais clara
- [ ] **Analise de threshold de decisao** (~2h) - Ajustar limite de classificacao para aumentar sensitivity (detectar mais casos positivos)

### Dissertacao (Escrita)
- [ ] **Secao "Limitacoes do Estudo"** (~2h) - Dataset pequeno (287 amostras), missing values (22.6%), desbalanceamento, resultados moderados (Kappa=0.33)
- [ ] **Comparacao com estado da arte** (~5h) - Revisar literatura: outros trabalhos conseguiram quais resultados em GAD/SAD? Baseline da area?
- [ ] **Discussao clinica aprofundada** (~4h) - Custo-beneficio de falsos negativos, viabilidade pratica, proposta de uso real
- [ ] **Interpretacao dos resultados medíocres** (~2h) - Por que Kappa=0.33? Por que Sensitivity=38%? Justificar ou reconhecer limitacoes

---

## 🟡 IMPORTANTE - Melhora Muito a Dissertacao

### Otimizacao de Modelos
- [ ] **GridSearch de hiperparametros** (~3h) - Otimizar C, gamma (SVM) e n_estimators, max_depth (XGBoost). Pode melhorar resultados
- [ ] **Learning Curves** (~2h) - Verificar se mais dados melhorariam o modelo. Diagnosticar overfitting/underfitting
- [ ] **Calibracao de probabilidades** (~2h) - Verificar se P(classe=1) realmente corresponde a probabilidade real (critico para decisoes clinicas)

### Analise Exploratoria de Dados (EDA)
- [ ] **Graficos de distribuicao das features** (~3h) - Visualizar distribuicao das 17 features, identificar outliers e padroes
- [ ] **Analise de multicolinearidade** (~2h) - VIF (Variance Inflation Factor) ou matriz de correlacao. Features redundantes?
- [ ] **Boxplots GAD vs nao-GAD** (~2h) - Comparar distribuicoes das features importantes entre classes
- [ ] **Analise das amostras SMOTE** (~2h) - Validar se amostras sinteticas sao realistas e plausíveis

### Fairness e Vies
- [ ] **Analise de vies racial** (~2h) - Race e importante (11%). Modelo e justo para todas racas? Sensitivity igual?
- [ ] **Analise de vies de genero** (~2h) - Sex e importante (5.4%). Modelo funciona igual para meninos e meninas?
- [ ] **Analise de vies de idade** (~2h) - Modelo funciona igual para todas faixas etarias? Learning curves por idade?
- [ ] **Metricas de fairness** (~3h) - Disparate impact, equalized odds, demographic parity

### Interpretabilidade
- [ ] **Visualizar arvore ADTree** (~2h) - Mostrar estrutura da arvore. Por que usar ADTree? Resposta: e interpretavel (mostrar a arvore!)
- [ ] **SHAP values** (~3h) - Explicar predicoes individuais. Por que crianca X foi classificada como GAD? Quais features contribuiram?
- [ ] **Analise de interacao entre features** (~3h) - Features importantes interagem? Ex: Age + Sex podem ter efeito combinado

---

## 🟢 DESEJAVEL - Diferenciais que Impressionam

### Validacao Avancada
- [ ] **Nested Cross-Validation** (~3h) - CV externo + interno para tuning de hiperparametros sem vazamento de informacao
- [ ] **Bootstrapping** (~2h) - Alternativa ao CV. Estimativas mais robustas de IC
- [ ] **Validacao estratificada por idade/sexo** (~3h) - Garantir representatividade em todos os folds

### Algoritmos Adicionais (Opcional)
- [ ] **Random Forest** (~2h) - Ensemble de arvores. Feature importance alternativo
- [ ] **Logistic Regression** (~1h) - Baseline simples e interpretavel
- [ ] **Ensemble de modelos** (~3h) - Combinar XGBoost + SVM + ADTree via voting ou stacking

### Analises Extras
- [ ] **Analise de sensibilidade** (~3h) - Como resultados mudam se remover features menos importantes?
- [ ] **Analise temporal** (~2h) - Dataset tem viés temporal? Criancas avaliadas em periodos diferentes?
- [ ] **Analise de missing values** (~2h) - Padroes nos missings? Missing at random ou informativo?
- [ ] **Feature engineering** (~4h) - Criar features derivadas. Ex: ratio de impairments/idade, interacoes

---

## 📊 Analise de Dados (Melhorar Conhecimento do Dataset)

### Pre-processamento Detalhado
- [ ] **Documentar tratamento de missing values** (~1h) - Como foram tratados os 22.6% de missings? Imputacao? Remocao?
- [ ] **Analise de outliers** (~2h) - Identificar e decidir o que fazer com valores extremos
- [ ] **Analise de distribuicoes** (~2h) - Features seguem distribuicao normal? Transformacoes necessarias?

### Estatistica Descritiva
- [ ] **Tabela 1 - Caracteristicas da amostra** (~2h) - Media±DP de cada feature, estratificada por GAD/SAD. Padrao em artigos medicos
- [ ] **Testes de associacao univariada** (~2h) - Qui-quadrado ou t-test para cada feature vs GAD/SAD. Quais sao significativas (p<0.05)?
- [ ] **Analise de prevalencia** (~1h) - Prevalencia de GAD/SAD por idade, sexo, raca. Contexto epidemiologico

---

## 📝 Dissertacao (Texto e Contexto)

### Introducao e Revisao
- [ ] **Contextualizar GAD/SAD** (~3h) - Prevalencia, impacto, custos, importancia do diagnostico precoce
- [ ] **Revisar ML em psiquiatria infantil** (~5h) - Estado da arte, metodos usados, resultados tipicos
- [ ] **Justificar escolha de features** (~2h) - Por que essas 17 features? Embasamento teorico/clinico

### Metodologia
- [ ] **Detalhar pre-processamento** (~2h) - Descrever cada etapa: limpeza, normalizacao, encoding
- [ ] **Justificar escolha de algoritmos** (~2h) - Por que ADTree, XGBoost, SVM? Vantagens de cada um
- [ ] **Descrever tecnicas de balanceamento** (~2h) - O que e SMOTE? Undersampling? Class weighting? Por que usar?

### Resultados
- [ ] **Tabelas formatadas (estilo JAMA/Lancet)** (~3h) - Seguir padrao de revistas medicas de alto impacto
- [ ] **Graficos profissionais** (~4h) - Melhorar visualizacoes. ROC curves, confusion matrices, feature importance
- [ ] **Reportar todos os modelos** (~2h) - Nao so o melhor. Mostrar todos os 12 modelos (3 algoritmos × 4 tecnicas)

### Discussao
- [ ] **Interpretar Kappa moderado** (~2h) - Por que 0.33? Comparar com literatura. E aceitavel? Limitacoes dos dados?
- [ ] **Discutir Sensitivity baixo 38%** (~2h) - Implicacoes clinicas de perder 62% dos casos. Trade-off Sens/Spec
- [ ] **Proposta de uso pratico** (~3h) - Como usar na clinica? Screening inicial? Ferramenta de apoio? Interface?
- [ ] **Consideracoes eticas** (~2h) - Privacidade, consentimento, risco de estigmatizacao, uso responsavel de IA

### Conclusao
- [ ] **Sumarizar contribuicoes** (~1h) - O que este trabalho adiciona ao campo? Metodologia? Insights?
- [ ] **Trabalhos futuros** (~1h) - Mais dados, mais features, deep learning, validacao externa

---

## Estrutura de Arquivos

```
scripts/
├── utils.py                  # Funcoes compartilhadas (metricas, IC, plots)
├── normalizacao.py           # Carrega e normaliza dados
├── modelo_adtree.py          # ADTree com 4 tecnicas
├── modelo_xgboost.py         # XGBoost com 4 tecnicas + Feature Importance
├── modelo_svm.py             # SVM com 4 tecnicas
├── comparativo_algoritmos.py # Compara ADTree vs XGBoost vs SVM
├── index.py                  # Analise exploratoria inicial
├── correlation.py            # Matriz de correlacao
└── compare_columns.py        # Comparacao de colunas treino/teste

output/plots/
├── ADtree/
│   └── GAD/                  # Resultados ADTree para GAD
│       └── plots/            # Graficos GAD
├── XGBoost/
│   ├── GAD/                  # Resultados XGBoost para GAD
│   │   └── plots/            # Graficos + Feature Importance GAD
│   └── SAD/                  # Resultados XGBoost para SAD
│       └── plots/            # Graficos + Feature Importance SAD
├── SVM/
│   ├── GAD/                  # Resultados SVM para GAD
│   │   └── plots/            # Graficos GAD
│   └── SAD/                  # Resultados SVM para SAD
│       └── plots/            # Graficos SAD
└── Comparativo/
    ├── GAD/                  # Comparativo para GAD
    │   └── plots/            # Graficos comparativos GAD
    └── SAD/                  # Comparativo para SAD
        └── plots/            # Graficos comparativos SAD
```

---

## 💀 Perguntas Criticas da Banca (Prepare-se!)

### Resultados
1. **"Por que o Kappa e apenas 0.33? Isso nao e muito baixo?"**
   - Precisa: Comparacao com literatura + discussao de limitacoes do dataset + reconhecer que e moderado

2. **"Sensitivity de 38% significa que voce perde 62% das criancas com GAD. Como justifica isso?"**
   - Precisa: Analise de threshold + discussao de trade-off Sens/Spec + proposta de uso como screening (nao diagnostico definitivo)

3. **"XGBoost e estatisticamente melhor que SVM?"**
   - Precisa: Teste de McNemar ou Wilcoxon (ICs se sobrepõem sem teste)

4. **"Onde esta a curva ROC?"**
   - Precisa: Implementar URGENTE (padrao em saude)

### Metodologia
5. **"Voce otimizou os hiperparametros?"**
   - Resposta atual: Nao. Precisa: GridSearch OU justificar valores (literatura, heuristica)

6. **"Por que nao usou o conjunto de teste holdout?"**
   - Precisa: Usar as 307 amostras de teste para validacao final

7. **"As amostras SMOTE sao realistas?"**
   - Precisa: Analise de plausibilidade das amostras sinteticas

8. **"Como tratou os 22.6% de missing values?"**
   - Precisa: Documentar tratamento (imputacao? remocao?) + justificar

### Interpretacao
9. **"Quais criancas o modelo erra mais? Por que?"**
   - Precisa: Analise de erros detalhada

10. **"Esse modelo e justo para todas racas/generos?"**
    - Precisa: Analise de fairness/vies

11. **"Feature X e importante, mas o que isso significa clinicamente?"**
    - Precisa: Interpretacao clinica de cada feature importante + consultar especialista

12. **"Por que usar ADTree se XGBoost e melhor?"**
    - Resposta: Interpretabilidade (mostrar arvore) OU remover ADTree do trabalho

### Aplicabilidade
13. **"Como esse modelo seria usado na pratica?"**
    - Precisa: Proposta de uso clinico + interface + viabilidade

14. **"Qual a contribuicao deste trabalho?"**
    - Precisa: Comparacao com literatura + identificar o que e novo/diferente

15. **"Quais as limitacoes do estudo?"**
    - Precisa: Secao completa sobre limitacoes (dataset pequeno, missing, desbalanceamento, generalizacao)

---

## 📈 Estimativa de Tempo Total

### URGENTE (24-28 horas)
- ROC + validacao teste + testes estatisticos + analise erros: ~10h
- Dissertacao (limitacoes + comparacao literatura + discussao): ~14h

### IMPORTANTE (20-25 horas)
- GridSearch + Learning curves + EDA + Fairness: ~15h
- Interpretabilidade + analise features: ~10h

### DESEJAVEL (15-20 horas)
- Nested CV + algoritmos extras + analises avancadas

**Total para nivel "bom/otimo": 44-53 horas de trabalho adicional**
**Total para nivel "aceitavel": 24-28 horas (apenas urgente)**

---

## 🎯 Plano de Acao Recomendado

### Semana 1 (Prioridade MAXIMA)
- [ ] Dia 1-2: ROC + AUC + validacao no teste
- [ ] Dia 3: Testes estatisticos + analise de erros
- [ ] Dia 4-5: Secao limitacoes + revisao literatura

### Semana 2 (Importante)
- [ ] Dia 1-2: GridSearch + Learning curves
- [ ] Dia 3-4: EDA completa + visualizacoes
- [ ] Dia 5: Analise de fairness + vies

### Semana 3 (Polimento)
- [ ] Dia 1-2: Interpretabilidade (SHAP, arvore ADTree)
- [ ] Dia 3-4: Discussao clinica + proposta de uso
- [ ] Dia 5: Revisao final + preparacao da apresentacao

---

## Notas para Defesa (Atualizadas)

> **Status Atual do Trabalho:**
> - ✅ Codigo bem estruturado e reproduzivel
> - ✅ Metodologia de CV correta (sem data leakage)
> - ✅ Multiplos algoritmos (ADTree, XGBoost, SVM)
> - ✅ 4 tecnicas de balanceamento testadas
> - ✅ Feature importance implementado
> - ❌ **Resultados moderados** (Kappa=0.33, Sens=38%)
> - ❌ **FALTA: ROC, validacao teste, testes estatisticos**
> - ❌ **FALTA: Discussao clinica aprofundada**
>
> **Nivel de Preparacao: 60%**
> **Nota Estimada Atual: 6.5-7.0 / 10**
> **Nota Potencial com Ajustes: 8.5-9.0 / 10**

---

## � Contexto dos Resultados (Entenda o que Voce Tem)

### Metricas Atuais (Melhor Modelo: XGBoost + SMOTE)
```
GAD:
- Accuracy: 85.0% (mas engana - dataset tem 84.7% de negativos)
- Sensitivity: 34.5% (PROBLEMA: detecta apenas 1 em cada 3 criancas com GAD)
- Specificity: 94.3% (bom - poucos falsos positivos)
- Kappa: 0.307 (concordancia "razoavel" mas no limite inferior)

SAD:
- Resultados similares (Kappa=0.256)
```

### O que os Numeros Significam?
- **Kappa 0.33**: Na escala de Landis & Koch, e "razoavel" (0.21-0.40), mas proximo de "leve"
- **Sensitivity 38%**: Em cada 100 criancas COM GAD, o modelo detecta apenas 38
- **62 criancas ficam sem diagnostico** = GRAVE em contexto clinico
- **Specificity 94%**: Em cada 100 criancas SEM GAD, o modelo acerta 94 (apenas 6 falsos alarmes)

### Por que os Resultados sao Moderados?
1. **Dataset pequeno**: 287 amostras para 17 features (marginalmente aceitavel)
2. **Desbalanceamento severo**: 84.7% vs 15.3% (apenas 44 casos positivos)
3. **22.6% missing values**: Muita informacao faltando
4. **Problema complexo**: GAD/SAD tem causas multifatoriais e subjetividade diagnostica

### Comparacao com Literatura (Pesquisar!)
- Trabalhos similares em ML para ansiedade infantil conseguem Kappa de quanto?
- Se literatura tem Kappa ~0.30-0.40: voce esta na media (ok mas nao excelente)
- Se literatura tem Kappa >0.50: seus resultados estao abaixo (precisa explicar por que)

---

## 🎯 Decisoes Estrategicas para Dissertacao

### Opcao A: Foco em "Proof of Concept" (Mais Seguro)
**Posicionamento:** "Este trabalho demonstra a **viabilidade** de usar ML para triagem de GAD/SAD,
mas reconhece limitacoes e propoe melhorias futuras"

**Vantagens:**
- Mais honesto e defensavel
- Banca valoriza reconhecimento de limitacoes
- Abre caminho para trabalhos futuros

**Desvantagens:**
- Nao pode afirmar "modelo pronto para uso clinico"

### Opcao B: Foco em "Ferramenta de Apoio" (Mais Ambicioso)
**Posicionamento:** "Sistema de **apoio a decisao** para screening inicial.
Nao substitui avaliacao clinica, mas auxilia identificacao de casos de risco"

**Vantagens:**
- Mais aplicabilidade pratica
- Sensitivity baixo e menos critico se houver avaliacao posterior
- Mais impacto potencial

**Desvantagens:**
- Banca vai cobrar mais validacao e analise clinica
- Precisa proposta de uso real + interface + validacao

### Recomendacao
**Opcao A inicialmente**, evoluir para B se melhorar resultados com GridSearch/mais analises

---

## �🔗 Referencias Importantes

### Metricas e Validacao
- Landis & Koch (1977) - Escala de interpretacao do Kappa
- Hanley & McNeil (1982) - ROC curves em medicina
- DeLong et al. (1988) - Comparacao de curvas ROC

### Machine Learning em Saude
- Choi et al. (2016) - ML para predicao de transtornos mentais
- Dwyer et al. (2018) - ML em psiquiatria infantil
- Bzdok & Meyer-Lindenberg (2018) - ML em neurociencia

### Fairness e Vies
- Obermeyer et al. (2019) - Vies racial em algoritmos de saude
- Rajkomar et al. (2018) - ML clinico e suas armadilhas

---

## ⚡ Comandos Rapidos (Atalhos Uteis)

### Executar Modelos
```bash
# Executar todos os modelos para GAD e SAD
python scripts/modelo_xgboost.py
python scripts/modelo_svm.py
python scripts/modelo_adtree.py
python scripts/comparativo_algoritmos.py

# Analise exploratoria
python scripts/correlation.py
python scripts/index.py
```

### Verificar Resultados
```bash
# Ver estrutura de pastas
find output/plots -type d | sort

# Contar arquivos gerados
find output/plots -name "*.txt" | wc -l
find output/plots -name "*.png" | wc -l

# Ver metricas do melhor modelo
cat output/plots/XGBoost/GAD/comparativo_gad.txt
cat output/plots/Comparativo/GAD/comparativo_algoritmos_gad.txt
```

### Analise do Dataset
```bash
cd datasets
python3 -c "
import pandas as pd
df = pd.read_csv('mestrado-treino.csv')
print(f'Treino: {df.shape}')
print(f'GAD: {df[\"GAD\"].value_counts().to_dict()}')
print(f'SAD: {df[\"SAD\"].value_counts().to_dict()}')
print(f'Missing: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100:.1f}%)')
"
```

---

## 🏁 Checklist Final Antes da Banca

### Codigo e Resultados
- [ ] Todos os scripts executam sem erro
- [ ] Curva ROC implementada e plotada
- [ ] Validacao no teste holdout executada
- [ ] Testes estatisticos executados (p-values calculados)
- [ ] Analise de erros documentada
- [ ] Feature importance interpretada clinicamente

### Dissertacao (Texto)
- [ ] Introducao contextualiza GAD/SAD (epidemiologia, impacto)
- [ ] Revisao bibliografica completa (min 30-40 referencias)
- [ ] Metodologia detalhada (pre-processamento, algoritmos, validacao)
- [ ] Resultados com tabelas formatadas (estilo JAMA)
- [ ] Discussao aprofundada (interpretacao + comparacao literatura)
- [ ] Secao "Limitacoes" completa e honesta
- [ ] Conclusao sumariza contribuicoes + trabalhos futuros

### Apresentacao
- [ ] Slides preparados (15-20 min de apresentacao)
- [ ] Graficos principais (ROC, confusion matrix, feature importance)
- [ ] Ensaiar respostas para 15 perguntas criticas acima
- [ ] Preparar demo/exemplo pratico (opcional mas impressiona)

### Documentacao
- [ ] README.md com instrucoes de reproducao
- [ ] Requirements.txt com versoes exatas das bibliotecas
- [ ] Scripts comentados e legíveis
- [ ] Dados anonimizados (verificar etica/privacidade)

---

## 🎓 Mensagem Final

**Voce tem um trabalho solido mas que precisa de refinamento.**

**Pontos Fortes:**
- Codigo bem estruturado ✅
- Metodologia correta (CV, sem data leakage) ✅
- Multiplos algoritmos e tecnicas ✅

**Proximos Passos Criticos:**
1. ROC + validacao teste + testes estatisticos (10h)
2. Dissertacao: limitacoes + literatura + discussao (14h)
3. Analise de erros + interpretacao clinica (5h)

**Total minimo para aprovar bem: ~30 horas**

**Lembre-se:** A banca valoriza mais **honestidade** sobre limitacoes do que
resultados perfeitos mal justificados. Seja critico com seu proprio trabalho
antes que eles sejam.

**Boa sorte! 🚀**
