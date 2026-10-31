"""
Treina um probe (classificador linear) em cima das ativações extraídas,
para prever grounded (1) vs confabulado (0) a partir do hidden state,
sem olhar para o texto gerado.

Duas formas de avaliação, porque medem coisas diferentes:

1. split aleatório (in-domain): quão separável é o sinal, no melhor caso.
2. split por categoria (out-of-domain): treina em categorias A, testa em
   categorias nunca vistas B. É isto que testa se a direção "grounded vs
   confabulado" é uma propriedade geral do modelo ou um artefacto do
   domínio de treino. É a pergunta em aberto mencionada na conversa.
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_layer(npz_path: str, layer: int):
    data = np.load(npz_path, allow_pickle=True)
    X = data[f"layer_{layer}"]
    y = data["labels"]
    cats = data["categories"]
    if "prompts" not in data.files:
        raise KeyError(
            f"{npz_path} has no 'prompts' array. It was written by a version of "
            "extraction.py that did not save it, and without it the in-domain "
            "split cannot be grouped by question. Re-run extraction.py."
        )
    return X, y, cats, data["prompts"]


def _fit_auc(X_tr, y_tr, X_te, y_te, seed=42):
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    return {
        "acc": accuracy_score(y_te, clf.predict(X_te)),
        "auc": roc_auc_score(y_te, probs),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "clf": clf,
    }


def eval_grouped_split(X, y, prompts, seed: int = 42, test_size: float = 0.2,
                        n_train: int | None = None):
    """
    In-domain score with the split grouped by question.

    Both claims for a question go to the same side of the split. This is the
    in-domain number to report: it still mixes categories freely, so it measures
    separability in the easy case, but it no longer lets the probe recognise a
    question it was trained on.

    n_train trims the training set to a fixed size, so this can be compared
    against the cross-category score at equal n.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=prompts))

    if n_train is not None and n_train < len(train_idx):
        rng = np.random.RandomState(seed)
        train_idx = rng.choice(train_idx, size=n_train, replace=False)

    return _fit_auc(X[train_idx], y[train_idx], X[test_idx], y[test_idx], seed)


def eval_random_split(X, y, seed: int = 42):
    """
    The original ungrouped split, kept only to measure how much it inflated the
    score. It splits on rows, so the grounded claim of a question can train the
    probe that is then tested on the confabulated claim of the same question:
    82.9% of test rows share their prompt with a training row on the TruthfulQA
    build. Do not report this number as the in-domain result.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return _fit_auc(X_tr, y_tr, X_te, y_te, seed)


def eval_cross_category(X, y, cats, train_cats: list[str], test_cats: list[str]):
    train_mask = np.isin(cats, train_cats)
    test_mask = np.isin(cats, test_cats)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("categorias de treino ou teste sem exemplos suficientes")

    res = _fit_auc(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
    res.pop("clf", None)
    return res


def mean_diff_direction(X, y):
    """Vetor direção grounded->confabulado, usado depois para steering."""
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="se omitido, tenta todas as layers guardadas no npz")
    args = ap.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    layers = args.layers or sorted(
        int(k.split("_")[1]) for k in data.files if k.startswith("layer_")
    )

    print(f"{'layer':>6} | {'auc (grouped)':>14} | {'auc (ungrouped)':>16} | {'inflation':>10}")
    for l in layers:
        X, y, cats, prompts = load_layer(args.activations, l)
        grouped = eval_grouped_split(X, y, prompts)
        leaky = eval_random_split(X, y)
        print(f"{l:>6} | {grouped['auc']:>14.3f} | {leaky['auc']:>16.3f} | "
              f"{leaky['auc'] - grouped['auc']:>+10.3f}")
