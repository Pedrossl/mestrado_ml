#!/usr/bin/env python3
"""
Compara colunas entre dois CSVs e imprime colunas em comum e colunas únicas.
Uso: python scripts/compare_columns.py [caminho_treino] [caminho_teste]
"""
import sys
import csv
import re
import difflib
import json
from collections import defaultdict

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


def read_headers(path):
    if _HAS_PANDAS:
        try:
            return pd.read_csv(path, nrows=0).columns.str.strip().tolist()
        except Exception as e:
            sys.stderr.write(f"Erro ao ler {path} com pandas: {e}\n")
            raise
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            return [h.strip() for h in next(reader)]


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/mestrado-treino.csv"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/mestrado-teste.csv"

    try:
        a = read_headers(train_path)
    except FileNotFoundError:
        sys.stderr.write(f"Arquivo não encontrado: {train_path}\n")
        sys.exit(1)

    try:
        b = read_headers(test_path)
    except FileNotFoundError:
        sys.stderr.write(f"Arquivo não encontrado: {test_path}\n")
        sys.exit(1)

    set_a = set(a)
    set_b = set(b)

    common = sorted(list(set_a & set_b))
    only_train = sorted(list(set_a - set_b))
    only_test = sorted(list(set_b - set_a))

    print(f"\n--- COMUNS ({len(common)}) ---")
    for c in common:
        print(c)

    print(f"\n--- APENAS NO TREINO ({len(only_train)}) ---")
    for c in only_train:
        print(c)

    print(f"\n--- APENAS NO TESTE ({len(only_test)}) ---")
    for c in only_test:
        print(c)

    # salvar resumo
    try:
        import json
        out = {
            "common": common,
            "only_train": only_train,
            "only_test": only_test,
        }
        with open("output/columns_comparison.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\nResumo salvo em: output/columns_comparison.json")
    except Exception as e:
        sys.stderr.write(f"Falha ao salvar resumo: {e}\n")


def normalize(name: str) -> str:
    """Normaliza um nome de coluna: minúsculas, remove pontuação e múltiplos espaços."""
    if name is None:
        return ""
    s = name.lower()
    s = re.sub(r"[\"'`\(\)\[\]\{\},\/\\\\:;\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


def similarity_score(a: str, b: str) -> float:
    """Combina SequenceMatcher ratio com Jaccard de tokens para uma pontuação 0..1."""
    na = normalize(a)
    nb = normalize(b)
    # fuzzy ratio (sequence)
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    # token overlap
    jacc = token_jaccard(na, nb)
    # dar mais peso ao token jaccard para capturar palavras semelhantes em ordens diferentes
    return 0.5 * ratio + 0.5 * jacc


def find_similar_columns(a_cols, b_cols, threshold=0.7, top_n=1):
    """Encontra pares semelhantes entre duas listas de colunas.

    Retorna lista de tuplas (col_a, col_b, score) ordenadas por score desc.
    """
    results = []
    for ca in a_cols:
        best = []
        for cb in b_cols:
            score = similarity_score(ca, cb)
            if score >= threshold:
                best.append((ca, cb, score))
        # manter apenas os top_n matches por coluna
        best_sorted = sorted(best, key=lambda x: x[2], reverse=True)[:top_n]
        results.extend(best_sorted)

    # também considerar matches começando do outro lado para pegar correspondências perdidas
    for cb in b_cols:
        for ca in a_cols:
            score = similarity_score(ca, cb)
            if score >= threshold:
                results.append((ca, cb, score))

    # dedup por par (ca,cb) e ordenar
    seen = set()
    uniq = []
    for ca, cb, score in sorted(results, key=lambda x: x[2], reverse=True):
        key = (ca, cb)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((ca, cb, round(score, 3)))
    return uniq


def suggest_and_save_matches(train_path, test_path, threshold=0.7):
    try:
        a = read_headers(train_path)
        b = read_headers(test_path)
    except Exception as e:
        raise

    matches = find_similar_columns(a, b, threshold=threshold, top_n=2)

    # Agrupar por coluna do treino
    grouped = defaultdict(list)
    for ca, cb, score in matches:
        grouped[ca].append({"test_col": cb, "score": score})

    suggestion = {"threshold": threshold, "matches": grouped}

    # salvar em JSON legível
    try:
        with open("output/columns_match_suggested.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in suggestion.items()}, f, ensure_ascii=False, indent=2, default=list)
        print("\nSugestões de mapeamento salvas em: output/columns_match_suggested.json")
    except Exception as e:
        sys.stderr.write(f"Falha ao salvar sugestões: {e}\n")

    # imprimir resumo
    if not matches:
        print("\nNenhuma coluna parecida encontrada com o limiar especificado.")
    else:
        print(f"\n--- SUGESTÕES DE COLUNAS PARECIDAS (threshold={threshold}) ---")
        for ca, cb, score in matches:
            print(f"{ca}  <-->  {cb}   (score={score})")


if __name__ == '__main__':
    # Comportamento adicional: sugerir matches similares
    train_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/mestrado-treino.csv"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/mestrado-teste.csv"
    try:
        suggest_and_save_matches(train_path, test_path, threshold=0.68)
    except Exception:
        pass


if __name__ == '__main__':
    main()
