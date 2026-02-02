import pandas as pd
import numpy as np
import logging
import os

# Desabilitar logs do Weka
logging.getLogger("weka").setLevel(logging.ERROR)
os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.util.logging.config.file=/dev/null"

import weka.core.jvm as jvm
from weka.core.dataset import Instances, Attribute, Instance
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

from normalizacao import carregar_teste_normalizado

OUTPUT_PATH = 'output/plots/ADtree'
PLOTS_PATH = 'output/plots/ADtree/plots'


def preparar_dados(target='GAD'):
    """
    Prepara os dados para o modelo ADTree.

    Args:
        target: Variável alvo ('GAD' ou 'SAD')

    Returns:
        DataFrame preparado, nome do target
    """
    df = carregar_teste_normalizado()

    # Remover colunas não preditoras
    colunas_remover = [
        'Subject',
        'GAD Probabiliy - Gamma',
        'SAD Probability - Gamma',
        'Sample Weight'
    ]

    if target == 'GAD':
        colunas_remover.append('SAD')
    else:
        colunas_remover.append('GAD')

    # Preparar features
    df_modelo = df.drop(columns=[c for c in colunas_remover if c in df.columns])

    # Converter Sex para numérico
    if 'Sex' in df_modelo.columns:
        df_modelo['Sex'] = df_modelo['Sex'].map({'M': 0, 'F': 1})

    # Garantir que target é a última coluna
    cols = [c for c in df_modelo.columns if c != target] + [target]
    df_modelo = df_modelo[cols]

    # Remover linhas com valores faltantes
    df_modelo = df_modelo.dropna()

    return df_modelo, target


def criar_dataset_weka(df):
    """Cria dataset Weka a partir de DataFrame com atributos nominais."""
    from weka.core.dataset import create_instances_from_lists

    # Criar atributos
    atts = []
    for col in df.columns:
        valores_unicos = sorted(df[col].dropna().unique())
        valores_str = [str(int(v)) if v == int(v) else str(v) for v in valores_unicos]
        atts.append(Attribute.create_nominal(col, valores_str))

    # Criar instâncias
    dataset = Instances.create_instances("dados_teste", atts, len(df))

    # Adicionar dados
    for _, row in df.iterrows():
        values = []
        for i, col in enumerate(df.columns):
            val = row[col]
            valores_unicos = sorted(df[col].dropna().unique())
            valores_str = [str(int(v)) if v == int(v) else str(v) for v in valores_unicos]
            val_str = str(int(val)) if val == int(val) else str(val)
            values.append(valores_str.index(val_str))
        inst = Instance.create_instance(values)
        dataset.add_instance(inst)

    dataset.class_is_last()
    return dataset


def exibir_resultados(evaluation, df, target_name, n_folds, n_treino, n_teste, modelo_nome, output_file):
    """Exibe e salva os resultados da avaliação."""
    # Extrair matriz de confusão
    cm = evaluation.confusion_matrix
    vn, fp = int(cm[0][0]), int(cm[0][1])
    fn, vp = int(cm[1][0]), int(cm[1][1])
    total = vn + fp + fn + vp

    # Calcular todas as métricas
    accuracy = evaluation.percent_correct
    sensitivity = vp / (vp + fn) if (vp + fn) > 0 else 0
    specificity = vn / (vn + fp) if (vn + fp) > 0 else 0
    ppv = vp / (vp + fp) if (vp + fp) > 0 else 0
    npv = vn / (vn + fn) if (vn + fn) > 0 else 0
    f1 = evaluation.f_measure(1)
    kappa = evaluation.kappa

    # Resultados
    print("\n" + "=" * 60)
    print("                      RESULTADOS")
    print("=" * 60)

    print("\n[MÉTRICAS DE DESEMPENHO]")
    print(f"  Accuracy (Acurácia):             {accuracy:>6.2f}%")
    print(f"  Sensitivity (Sensibilidade):    {sensitivity*100:>6.2f}%")
    print(f"  Specificity (Especificidade):   {specificity*100:>6.2f}%")
    print(f"  PPV - Precision (Precisão):     {ppv*100:>6.2f}%")
    print(f"  NPV (Valor Pred. Negativo):     {npv*100:>6.2f}%")
    print(f"  F1-Score:                       {f1*100:>6.2f}%")
    print(f"  Kappa:                          {kappa:>6.4f}")

    # Matriz de confusão
    print("\n[MATRIZ DE CONFUSÃO]")
    print("")
    print("                       Modelo previu:")
    print(f"                    SEM {target_name}     COM {target_name}")
    print("                  +-----------+-----------+")
    print(f"  Paciente   SEM  |    {vn:^5}  |    {fp:^5}  |")
    print(f"  realmente       |    (VN)   |    (FP)   |")
    print("                  +-----------+-----------+")
    print(f"             COM  |    {fn:^5}  |    {vp:^5}  |")
    print(f"                  |    (FN)   |    (VP)   |")
    print("                  +-----------+-----------+")
    print("")
    print("  Acertos:")
    print(f"    VN = {vn} pacientes SEM {target_name}, modelo acertou (previu SEM)")
    print(f"    VP = {vp} pacientes COM {target_name}, modelo acertou (previu COM)")
    print("")
    print("  Erros:")
    print(f"    FP = {fp} pacientes SEM {target_name}, modelo errou (previu COM)")
    print(f"    FN = {fn} pacientes COM {target_name}, modelo errou (previu SEM)")

    # Resumo
    print("\n[RESUMO]")
    print(f"  Total: {total} | Acertos: {vn+vp} ({(vn+vp)/total*100:.1f}%) | Erros: {fp+fn} ({(fp+fn)/total*100:.1f}%)")

    # Salvar métricas
    print("\n" + "=" * 60)
    print("                  ARQUIVO SALVO")
    print("=" * 60)

    with open(output_file, 'w') as f:
        f.write(f"{modelo_nome} - {target_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Amostras: {df.shape[0]} | Features: {df.shape[1] - 1}\n")
        f.write(f"Validação: {n_folds}-fold CV (treino={n_treino}, teste={n_teste})\n\n")
        f.write("MÉTRICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:    {accuracy:.2f}%\n")
        f.write(f"Sensitivity: {sensitivity*100:.2f}%\n")
        f.write(f"Specificity: {specificity*100:.2f}%\n")
        f.write(f"PPV:         {ppv*100:.2f}%\n")
        f.write(f"NPV:         {npv*100:.2f}%\n")
        f.write(f"F1-Score:    {f1*100:.2f}%\n")
        f.write(f"Kappa:       {kappa:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO\n")
        f.write("-" * 40 + "\n")
        f.write(f"VN={vn} | FP={fp}\n")
        f.write(f"FN={fn} | VP={vp}\n\n")
        f.write(f"Total: {total} | Acertos: {vn+vp} | Erros: {fp+fn}\n")
    print(f"  Métricas: {output_file}")
    print("\n" + "=" * 60 + "\n")


def treinar_adtree(target='GAD'):
    """
    Treina modelo ADTree usando Weka (sem class weighting).

    Args:
        target: Variável alvo ('GAD' ou 'SAD')
    """
    # Preparar dados
    df, target_name = preparar_dados(target)

    # Header
    print("\n" + "=" * 60)
    print("         MODELO ADTree - SEM CLASS WEIGHTING")
    print("=" * 60)

    # Info do dataset
    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    # Criar dataset Weka
    print("\n[TREINAMENTO]")
    print("  Criando dataset Weka...", end=" ")
    dataset = criar_dataset_weka(df)
    print("OK")

    # Criar e treinar classificador
    print("  Treinando ADTree...", end=" ")
    adtree = Classifier(classname="weka.classifiers.trees.ADTree")
    adtree.options = ["-B", "10", "-E", "-3"]
    adtree.build_classifier(dataset)
    print("OK")

    # Avaliação com 10-fold cross-validation
    # Divisão: 90% treino (258 amostras) / 10% teste (29 amostras) por rodada
    n_folds = 10
    n_teste = len(df) // n_folds
    n_treino = len(df) - n_teste
    print(f"  Executando {n_folds}-fold CV (treino={n_treino}, teste={n_teste} por fold)...", end=" ")
    evaluation = Evaluation(dataset)
    evaluation.crossvalidate_model(adtree, dataset, n_folds, Random(1))
    print("OK")

    # Exibir resultados
    output_file = f'{OUTPUT_PATH}/adtree_{target.lower()}_metricas.txt'
    exibir_resultados(evaluation, df, target_name, n_folds, n_treino, n_teste,
                      "MODELO ADTree (sem weighting)", output_file)

    return adtree, evaluation


def treinar_adtree_weighted(target='GAD'):
    """
    Treina modelo ADTree com Class Weighting usando CostSensitiveClassifier.

    Args:
        target: Variável alvo ('GAD' ou 'SAD')
    """
    # Preparar dados
    df, target_name = preparar_dados(target)

    # Header
    print("\n" + "=" * 60)
    print("        MODELO ADTree - COM CLASS WEIGHTING")
    print("=" * 60)

    # Info do dataset
    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    peso = dist[0] / dist[1]  # Peso para classe minoritária
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Peso aplicado à classe 1: {peso:.2f}x")

    # Criar dataset Weka
    print("\n[TREINAMENTO]")
    print("  Criando dataset Weka...", end=" ")
    dataset = criar_dataset_weka(df)
    print("OK")

    # Criar ADTree base
    print("  Configurando ADTree com CostSensitiveClassifier...", end=" ")
    adtree = Classifier(classname="weka.classifiers.trees.ADTree")
    adtree.options = ["-B", "10", "-E", "-3"]

    # Criar CostSensitiveClassifier com matriz de custos
    # Matriz: [[custo_vn, custo_fp], [custo_fn, custo_vp]]
    # Queremos penalizar mais os FN (falsos negativos)
    cost_sensitive = Classifier(classname="weka.classifiers.meta.CostSensitiveClassifier")

    # Configurar matriz de custos: penaliza FN mais que FP
    # -cost-matrix "[0 1; peso 0]" significa custo=peso para FN
    cost_matrix = f"[0 1; {peso:.1f} 0]"
    cost_sensitive.options = [
        "-cost-matrix", cost_matrix,
        "-S", "1",  # Seed
        "-W", "weka.classifiers.trees.ADTree",
        "--", "-B", "10", "-E", "-3"
    ]

    cost_sensitive.build_classifier(dataset)
    print("OK")

    # Avaliação
    n_folds = 10
    n_teste = len(df) // n_folds
    n_treino = len(df) - n_teste
    print(f"  Executando {n_folds}-fold CV (treino={n_treino}, teste={n_teste} por fold)...", end=" ")
    evaluation = Evaluation(dataset)
    evaluation.crossvalidate_model(cost_sensitive, dataset, n_folds, Random(1))
    print("OK")

    # Exibir resultados
    output_file = f'{OUTPUT_PATH}/adtree_{target.lower()}_weighted_metricas.txt'
    exibir_resultados(evaluation, df, target_name, n_folds, n_treino, n_teste,
                      f"MODELO ADTree (com weighting {peso:.1f}x)", output_file)

    return cost_sensitive, evaluation


def treinar_adtree_smote(target='GAD'):
    """
    Treina modelo ADTree com SMOTE aplicado corretamente dentro de cada fold.
    Evita data leakage aplicando SMOTE apenas nos dados de treino.

    Args:
        target: Variável alvo ('GAD' ou 'SAD')
    """
    from sklearn.model_selection import StratifiedKFold

    # Preparar dados
    df, target_name = preparar_dados(target)

    # Header
    print("\n" + "=" * 60)
    print("       MODELO ADTree - COM SMOTE (CORRIGIDO)")
    print("=" * 60)

    # Info do dataset original
    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    # Separar features e target
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    # Cross-validation manual com SMOTE dentro de cada fold
    print("\n[CROSS-VALIDATION COM SMOTE POR FOLD]")
    print("  SMOTE aplicado apenas no treino (sem data leakage)")

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Acumuladores para métricas
    total_vp, total_vn, total_fp, total_fn = 0, 0, 0, 0

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Separar treino e teste
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Aplicar SMOTE apenas no treino
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Criar DataFrame para treino (com SMOTE)
        df_train = pd.DataFrame(X_train_resampled, columns=feature_names)
        df_train[target_name] = y_train_resampled

        # Criar DataFrame para teste (dados originais, sem SMOTE)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        # Criar datasets Weka
        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        # Treinar classificador
        adtree = Classifier(classname="weka.classifiers.trees.ADTree")
        adtree.options = ["-B", "10", "-E", "-3"]
        adtree.build_classifier(dataset_train)

        # Avaliar no teste
        evaluation = Evaluation(dataset_train)
        evaluation.test_model(adtree, dataset_test)

        # Acumular matriz de confusão
        cm = evaluation.confusion_matrix
        total_vn += int(cm[0][0])
        total_fp += int(cm[0][1])
        total_fn += int(cm[1][0])
        total_vp += int(cm[1][1])

    print("OK")

    # Calcular métricas agregadas
    total = total_vn + total_fp + total_fn + total_vp
    accuracy = (total_vn + total_vp) / total * 100
    sensitivity = total_vp / (total_vp + total_fn) * 100 if (total_vp + total_fn) > 0 else 0
    specificity = total_vn / (total_vn + total_fp) * 100 if (total_vn + total_fp) > 0 else 0
    ppv = total_vp / (total_vp + total_fp) * 100 if (total_vp + total_fp) > 0 else 0
    npv = total_vn / (total_vn + total_fn) * 100 if (total_vn + total_fn) > 0 else 0

    # F1-Score
    precision = total_vp / (total_vp + total_fp) if (total_vp + total_fp) > 0 else 0
    recall = total_vp / (total_vp + total_fn) if (total_vp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

    # Kappa
    po = (total_vn + total_vp) / total
    pe = ((total_vn + total_fp) * (total_vn + total_fn) + (total_fn + total_vp) * (total_fp + total_vp)) / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # Exibir resultados
    print("\n" + "=" * 60)
    print("                      RESULTADOS")
    print("=" * 60)

    print("\n[MÉTRICAS DE DESEMPENHO]")
    print(f"  Accuracy (Acurácia):             {accuracy:>6.2f}%")
    print(f"  Sensitivity (Sensibilidade):    {sensitivity:>6.2f}%")
    print(f"  Specificity (Especificidade):   {specificity:>6.2f}%")
    print(f"  PPV - Precision (Precisão):     {ppv:>6.2f}%")
    print(f"  NPV (Valor Pred. Negativo):     {npv:>6.2f}%")
    print(f"  F1-Score:                       {f1:>6.2f}%")
    print(f"  Kappa:                          {kappa:>6.4f}")

    # Matriz de confusão
    print("\n[MATRIZ DE CONFUSÃO AGREGADA]")
    print("")
    print("                       Modelo previu:")
    print(f"                    SEM {target_name}     COM {target_name}")
    print("                  +-----------+-----------+")
    print(f"  Paciente   SEM  |    {total_vn:^5}  |    {total_fp:^5}  |")
    print(f"  realmente       |    (VN)   |    (FP)   |")
    print("                  +-----------+-----------+")
    print(f"             COM  |    {total_fn:^5}  |    {total_vp:^5}  |")
    print(f"                  |    (FN)   |    (VP)   |")
    print("                  +-----------+-----------+")

    # Salvar métricas
    output_file = f'{OUTPUT_PATH}/adtree_{target.lower()}_smote_metricas.txt'
    with open(output_file, 'w') as f:
        f.write(f"MODELO ADTree (com SMOTE corrigido) - {target_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write("NOTA: SMOTE aplicado apenas no treino de cada fold\n")
        f.write("      para evitar data leakage.\n\n")
        f.write(f"Amostras originais: {df.shape[0]} | Features: {df.shape[1] - 1}\n")
        f.write(f"Validação: {n_folds}-fold Stratified CV\n\n")
        f.write("MÉTRICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:    {accuracy:.2f}%\n")
        f.write(f"Sensitivity: {sensitivity:.2f}%\n")
        f.write(f"Specificity: {specificity:.2f}%\n")
        f.write(f"PPV:         {ppv:.2f}%\n")
        f.write(f"NPV:         {npv:.2f}%\n")
        f.write(f"F1-Score:    {f1:.2f}%\n")
        f.write(f"Kappa:       {kappa:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO AGREGADA\n")
        f.write("-" * 40 + "\n")
        f.write(f"VN={total_vn} | FP={total_fp}\n")
        f.write(f"FN={total_fn} | VP={total_vp}\n\n")
        f.write(f"Total: {total} | Acertos: {total_vn+total_vp} | Erros: {total_fp+total_fn}\n")

    print(f"\n  Métricas salvas em: {output_file}")
    print("\n" + "=" * 60 + "\n")

    # Retornar objeto compatível com comparativo
    class ResultadoSMOTE:
        def __init__(self, acc, sens, spec, ppv, f1, kappa, cm):
            self.percent_correct = acc
            self.confusion_matrix = cm
            self.kappa = kappa
            self._f1 = f1 / 100

        def f_measure(self, idx):
            return self._f1

    cm_array = [[total_vn, total_fp], [total_fn, total_vp]]
    resultado = ResultadoSMOTE(accuracy, sensitivity, specificity, ppv, f1, kappa, cm_array)

    return None, resultado


def treinar_adtree_undersampling(target='GAD'):
    """
    Treina modelo ADTree com Undersampling aplicado corretamente dentro de cada fold.
    Evita data leakage aplicando Undersampling apenas nos dados de treino.

    Args:
        target: Variável alvo ('GAD' ou 'SAD')
    """
    from sklearn.model_selection import StratifiedKFold

    # Preparar dados
    df, target_name = preparar_dados(target)

    # Header
    print("\n" + "=" * 60)
    print("    MODELO ADTree - COM UNDERSAMPLING (CORRIGIDO)")
    print("=" * 60)

    # Info do dataset original
    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    # Separar features e target
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    # Cross-validation manual com Undersampling dentro de cada fold
    print("\n[CROSS-VALIDATION COM UNDERSAMPLING POR FOLD]")
    print("  Undersampling aplicado apenas no treino (sem data leakage)")

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Acumuladores para métricas
    total_vp, total_vn, total_fp, total_fn = 0, 0, 0, 0

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Separar treino e teste
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Aplicar Undersampling apenas no treino
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

        # Criar DataFrame para treino (com Undersampling)
        df_train = pd.DataFrame(X_train_resampled, columns=feature_names)
        df_train[target_name] = y_train_resampled

        # Criar DataFrame para teste (dados originais)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        # Criar datasets Weka
        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        # Treinar classificador
        adtree = Classifier(classname="weka.classifiers.trees.ADTree")
        adtree.options = ["-B", "10", "-E", "-3"]
        adtree.build_classifier(dataset_train)

        # Avaliar no teste
        evaluation = Evaluation(dataset_train)
        evaluation.test_model(adtree, dataset_test)

        # Acumular matriz de confusão
        cm = evaluation.confusion_matrix
        total_vn += int(cm[0][0])
        total_fp += int(cm[0][1])
        total_fn += int(cm[1][0])
        total_vp += int(cm[1][1])

    print("OK")

    # Calcular métricas agregadas
    total = total_vn + total_fp + total_fn + total_vp
    accuracy = (total_vn + total_vp) / total * 100
    sensitivity = total_vp / (total_vp + total_fn) * 100 if (total_vp + total_fn) > 0 else 0
    specificity = total_vn / (total_vn + total_fp) * 100 if (total_vn + total_fp) > 0 else 0
    ppv = total_vp / (total_vp + total_fp) * 100 if (total_vp + total_fp) > 0 else 0
    npv = total_vn / (total_vn + total_fn) * 100 if (total_vn + total_fn) > 0 else 0

    # F1-Score
    precision = total_vp / (total_vp + total_fp) if (total_vp + total_fp) > 0 else 0
    recall = total_vp / (total_vp + total_fn) if (total_vp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

    # Kappa
    po = (total_vn + total_vp) / total
    pe = ((total_vn + total_fp) * (total_vn + total_fn) + (total_fn + total_vp) * (total_fp + total_vp)) / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # Exibir resultados
    print("\n" + "=" * 60)
    print("                      RESULTADOS")
    print("=" * 60)

    print("\n[MÉTRICAS DE DESEMPENHO]")
    print(f"  Accuracy (Acurácia):             {accuracy:>6.2f}%")
    print(f"  Sensitivity (Sensibilidade):    {sensitivity:>6.2f}%")
    print(f"  Specificity (Especificidade):   {specificity:>6.2f}%")
    print(f"  PPV - Precision (Precisão):     {ppv:>6.2f}%")
    print(f"  NPV (Valor Pred. Negativo):     {npv:>6.2f}%")
    print(f"  F1-Score:                       {f1:>6.2f}%")
    print(f"  Kappa:                          {kappa:>6.4f}")

    # Matriz de confusão
    print("\n[MATRIZ DE CONFUSÃO AGREGADA]")
    print("")
    print("                       Modelo previu:")
    print(f"                    SEM {target_name}     COM {target_name}")
    print("                  +-----------+-----------+")
    print(f"  Paciente   SEM  |    {total_vn:^5}  |    {total_fp:^5}  |")
    print(f"  realmente       |    (VN)   |    (FP)   |")
    print("                  +-----------+-----------+")
    print(f"             COM  |    {total_fn:^5}  |    {total_vp:^5}  |")
    print(f"                  |    (FN)   |    (VP)   |")
    print("                  +-----------+-----------+")

    # Salvar métricas
    output_file = f'{OUTPUT_PATH}/adtree_{target.lower()}_undersampling_metricas.txt'
    with open(output_file, 'w') as f:
        f.write(f"MODELO ADTree (com Undersampling corrigido) - {target_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write("NOTA: Undersampling aplicado apenas no treino de cada fold\n")
        f.write("      para evitar data leakage.\n\n")
        f.write(f"Amostras originais: {df.shape[0]} | Features: {df.shape[1] - 1}\n")
        f.write(f"Validação: {n_folds}-fold Stratified CV\n\n")
        f.write("MÉTRICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:    {accuracy:.2f}%\n")
        f.write(f"Sensitivity: {sensitivity:.2f}%\n")
        f.write(f"Specificity: {specificity:.2f}%\n")
        f.write(f"PPV:         {ppv:.2f}%\n")
        f.write(f"NPV:         {npv:.2f}%\n")
        f.write(f"F1-Score:    {f1:.2f}%\n")
        f.write(f"Kappa:       {kappa:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO AGREGADA\n")
        f.write("-" * 40 + "\n")
        f.write(f"VN={total_vn} | FP={total_fp}\n")
        f.write(f"FN={total_fn} | VP={total_vp}\n\n")
        f.write(f"Total: {total} | Acertos: {total_vn+total_vp} | Erros: {total_fp+total_fn}\n")

    print(f"\n  Métricas salvas em: {output_file}")
    print("\n" + "=" * 60 + "\n")

    # Retornar objeto compatível com comparativo
    class ResultadoUndersampling:
        def __init__(self, acc, sens, spec, ppv, f1, kappa, cm):
            self.percent_correct = acc
            self.confusion_matrix = cm
            self.kappa = kappa
            self._f1 = f1 / 100

        def f_measure(self, idx):
            return self._f1

    cm_array = [[total_vn, total_fp], [total_fn, total_vp]]
    resultado = ResultadoUndersampling(accuracy, sensitivity, specificity, ppv, f1, kappa, cm_array)

    return None, resultado


def comparativo_modelos(avaliacoes, target_name):
    """
    Exibe tabela comparativa de todos os modelos.

    Args:
        avaliacoes: Dict com nome do modelo e tupla (evaluation, df_size)
        target_name: Nome do target
    """
    print("\n" + "=" * 80)
    print("                         COMPARATIVO DE MODELOS")
    print("=" * 80)

    # Header da tabela
    print(f"\n{'Modelo':<25} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}")
    print("-" * 80)

    resultados = []

    for nome, (evaluation, _) in avaliacoes.items():
        # Extrair métricas
        cm = evaluation.confusion_matrix
        vn, fp = int(cm[0][0]), int(cm[0][1])
        fn, vp = int(cm[1][0]), int(cm[1][1])

        accuracy = evaluation.percent_correct
        sensitivity = vp / (vp + fn) * 100 if (vp + fn) > 0 else 0
        specificity = vn / (vn + fp) * 100 if (vn + fp) > 0 else 0
        ppv = vp / (vp + fp) * 100 if (vp + fp) > 0 else 0
        f1 = evaluation.f_measure(1) * 100
        kappa = evaluation.kappa

        print(f"{nome:<25} {accuracy:>7.2f}% {sensitivity:>7.2f}% {specificity:>7.2f}% {ppv:>7.2f}% {f1:>7.2f}% {kappa:>8.4f}")

        resultados.append({
            'Modelo': nome,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'F1-Score': f1,
            'Kappa': kappa
        })

    print("-" * 80)

    # Identificar melhor modelo por métrica
    print("\n[MELHORES POR MÉTRICA]")
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    for metrica in metricas:
        melhor = max(resultados, key=lambda x: x[metrica])
        print(f"  {metrica:<12}: {melhor['Modelo']} ({melhor[metrica]:.2f}{'%' if metrica != 'Kappa' else ''})")

    # Salvar comparativo
    output_file = f'{OUTPUT_PATH}/comparativo_{target_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"COMPARATIVO DE MODELOS - {target_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Modelo':<25} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}\n")
        f.write("-" * 80 + "\n")
        for r in resultados:
            f.write(f"{r['Modelo']:<25} {r['Accuracy']:>7.2f}% {r['Sensitivity']:>7.2f}% {r['Specificity']:>7.2f}% {r['PPV']:>7.2f}% {r['F1-Score']:>7.2f}% {r['Kappa']:>8.4f}\n")
        f.write("-" * 80 + "\n\n")
        f.write("MELHORES POR MÉTRICA\n")
        f.write("-" * 40 + "\n")
        for metrica in metricas:
            melhor = max(resultados, key=lambda x: x[metrica])
            f.write(f"{metrica:<12}: {melhor['Modelo']} ({melhor[metrica]:.2f}{'%' if metrica != 'Kappa' else ''})\n")

    print(f"\n  Comparativo salvo em: {output_file}")
    print("\n" + "=" * 80 + "\n")

    return resultados


def plotar_comparativo(resultados, target_name):
    """
    Gera gráfico de barras e tabela comparativa dos modelos.

    Args:
        resultados: Lista de dicts com métricas de cada modelo
        target_name: Nome do target
    """
    # Dados
    modelos = [r['Modelo'] for r in resultados]
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    metricas_grafico = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score']

    # Cores para cada modelo
    cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # ========== GRÁFICO DE BARRAS ==========
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Posições das barras
    x = np.arange(len(metricas_grafico))
    largura = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    # Criar barras para cada modelo
    for i, (modelo, cor) in enumerate(zip(modelos, cores)):
        valores = [resultados[i][m] for m in metricas_grafico]
        barras = ax.bar(x + offsets[i] * largura, valores, largura,
                        label=modelo, color=cor, edgecolor='white', linewidth=0.5)

        # Adicionar valores nas barras
        for barra, valor in zip(barras, valores):
            altura = barra.get_height()
            ax.annotate(f'{valor:.1f}',
                        xy=(barra.get_x() + barra.get_width() / 2, altura),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    # Configurações do gráfico
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparativo de Modelos ADTree - {target_name}\nTécnicas de Balanceamento de Classes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_grafico, fontsize=11)
    ax.set_ylim(0, 105)

    # Linha de referência em 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legenda
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Criar pasta de plots se não existir
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Salvar gráfico
    grafico_file = f'{PLOTS_PATH}/comparativo_{target_name.lower()}_grafico.png'
    plt.savefig(grafico_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Gráfico salvo em: {grafico_file}")

    # ========== TABELA VISUAL ==========
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Preparar dados da tabela
    colunas = ['Modelo'] + metricas
    dados_tabela = []
    for r in resultados:
        linha = [r['Modelo']]
        for m in metricas:
            if m == 'Kappa':
                linha.append(f"{r[m]:.4f}")
            else:
                linha.append(f"{r[m]:.2f}%")
        dados_tabela.append(linha)

    # Criar tabela
    tabela = ax.table(
        cellText=dados_tabela,
        colLabels=colunas,
        cellLoc='center',
        loc='center',
        colColours=['#2c3e50'] + ['#34495e'] * len(metricas)
    )

    # Estilizar tabela
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1.2, 2)

    # Cores do header
    for j in range(len(colunas)):
        tabela[(0, j)].set_text_props(color='white', fontweight='bold')

    # Cores alternadas nas linhas e destaque do melhor valor
    for i, r in enumerate(resultados):
        # Cor de fundo alternada
        cor_fundo = cores[i] + '30'  # Adiciona transparência
        for j in range(len(colunas)):
            tabela[(i + 1, j)].set_facecolor(cor_fundo)
            tabela[(i + 1, j)].set_text_props(fontweight='normal')

    # Destacar melhores valores por coluna
    for j, metrica in enumerate(metricas):
        valores = [r[metrica] for r in resultados]
        melhor_idx = valores.index(max(valores))
        tabela[(melhor_idx + 1, j + 1)].set_text_props(fontweight='bold', color='#27ae60')

    # Título
    ax.set_title(f'Comparativo de Métricas - ADTree ({target_name})',
                 fontsize=14, fontweight='bold', pad=20, y=0.95)

    plt.tight_layout()

    # Salvar tabela
    tabela_file = f'{PLOTS_PATH}/comparativo_{target_name.lower()}_tabela.png'
    plt.savefig(tabela_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Tabela salva em: {tabela_file}")

    return grafico_file, tabela_file


def instalar_pacote_adtree():
    """Instala o pacote ADTree do Weka se não estiver instalado."""
    import weka.core.packages as packages

    pkg_name = "alternatingDecisionTrees"
    installed = [p.name for p in packages.installed_packages()]

    if pkg_name not in installed:
        print(f"Instalando pacote {pkg_name}...")
        packages.install_package(pkg_name)
        print("Pacote instalado! Reinicie o script.")
        return False
    return True


if __name__ == "__main__":
    # Suprimir logs do Weka
    import sys
    from io import StringIO

    # Iniciar JVM (capturando stderr temporariamente)
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    jvm.start(packages=True, logging_level=logging.ERROR)

    sys.stderr = old_stderr

    try:
        if instalar_pacote_adtree():
            target = 'GAD'
            avaliacoes = {}

            # Modelo 1: Sem Class Weighting
            modelo1, avaliacao1 = treinar_adtree(target)
            avaliacoes['Sem Balanceamento'] = (avaliacao1, None)

            # Separador
            print("\n" + "@" * 60)
            print("@" * 60 + "\n")

            # Modelo 2: Com Class Weighting
            modelo2, avaliacao2 = treinar_adtree_weighted(target)
            avaliacoes['Class Weighting'] = (avaliacao2, None)

            # Separador
            print("\n" + "@" * 60)
            print("@" * 60 + "\n")

            # Modelo 3: Com SMOTE
            modelo3, avaliacao3 = treinar_adtree_smote(target)
            avaliacoes['SMOTE'] = (avaliacao3, None)

            # Separador
            print("\n" + "@" * 60)
            print("@" * 60 + "\n")

            # Modelo 4: Com Undersampling
            modelo4, avaliacao4 = treinar_adtree_undersampling(target)
            avaliacoes['Undersampling'] = (avaliacao4, None)

            # Separador
            print("\n" + "@" * 60)
            print("@" * 60 + "\n")

            # Comparativo final
            resultados = comparativo_modelos(avaliacoes, target)

            # Gráfico comparativo
            plotar_comparativo(resultados, target)

    finally:
        jvm.stop()
