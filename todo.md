# TODO - Mestrado ML

## 🔴 Pendências Críticas

### Pré-processamento
- [ ] Trocar variável SEX para codificação binária (0 e 1)

### Revisão Bibliográfica
- [ ] Procurar mais artigos sobre ansiedade em crianças
- [ ] Definir contribuição do artigo
- [ ] Listar desvantagens da técnica ADTree

### Experimentos
- [ ] Comparar ADTree com outros algoritmos
  - [ ] Random Forest
  - [ ] SVM
  - [ ] XGBoost
  - [ ] Logistic Regression
  - [ ] J48 (C4.5)
  - [ ] Naive Bayes
- [ ] Testar também para SAD (além de GAD)
- [ ] Adicionar intervalos de confiança (desvio padrão entre folds)
- [ ] Seleção de features / Análise de importância
  - Quais variáveis mais contribuem para prever GAD?

---

## 📊 Algoritmos a Implementar

### Árvores e Ensemble
| Algoritmo | Descrição | Melhor para |
|-----------|-----------|-------------|
| J48 (C4.5) | Árvore de decisão clássica | Interpretabilidade |
| Random Forest | Múltiplas árvores votando juntas | Melhor precisão |
| Gradient Boosting | Árvores sequenciais corrigindo erros | Alta performance |

### Outros Classificadores
| Algoritmo | Descrição | Melhor para |
|-----------|-----------|-------------|
| Naive Bayes | Probabilístico simples | Datasets pequenos |
| SVM | Separação com hiperplano | Dados complexos |
| KNN | Classificação por vizinhos próximos | Simples e intuitivo |
| Logistic Regression | Regressão para classificação | Baseline interpretável |

### Técnicas para Desbalanceamento
| Técnica | Descrição |
|---------|-----------|
| SMOTE | Cria amostras sintéticas da classe minoritária |
| Class Weighting | Atribui maior peso à classe minoritária |
| Undersampling | Reduz amostras da classe majoritária |

---

## 🎯 Ordem Sugerida de Implementação

1. [ ] **J48** - Comparar com ADTree (ambas são árvores)
2. [ ] **Random Forest** - Geralmente melhora resultados
3. [ ] **Naive Bayes** - Rápido, bom baseline
4. [ ] **SMOTE + ADTree** - Tratar desbalanceamento
5. [ ] **SVM** - Para comparação com métodos não baseados em árvores

---

## 🔵 Análises Opcionais (Diferenciais)

- [ ] Gerar curvas ROC e calcular AUC
  - Métrica padrão em classificação médica
- [ ] Análise de erros
  - Quais pacientes o modelo erra mais? Por quê?
- [ ] Matriz de confusão detalhada
- [ ] Análise de correlação entre features

---

## 📝 Notas

> **Justificativa para a banca:** A comparação com múltiplos algoritmos demonstra que a escolha do ADTree foi baseada em análise comparativa rigorosa, não apenas conveniência.
