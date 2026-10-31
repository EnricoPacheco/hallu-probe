"""
Generalisation sweep. For each layer it reports three numbers:

- in-domain, with the split grouped by question. Categories mix freely, but
  both claims of a question stay on the same side of the split.
- cross-category, trained on one set of categories and tested on categories
  never seen, with the training set trimmed to the same size as the in-domain
  one so the gap is not partly a sample-size effect.
- the gap between the two, which is the quantity the project is about. A small
  gap is evidence that the grounded-versus-confabulated direction is a general
  property of the model; a large gap says the probe is reading the domain.

It also reports the ungrouped in-domain score, not as a result but to show how
much the original row-wise split inflated it.
"""

import argparse
import json

import numpy as np

from probe import (
    eval_cross_category,
    eval_grouped_split,
    eval_random_split,
    load_layer,
)


def run_sweep(npz_path: str, layers: list[int], holdout_frac: float = 0.3, seed: int = 42):
    rng = np.random.RandomState(seed)
    data = np.load(npz_path, allow_pickle=True)
    all_cats = sorted(set(data["categories"].tolist()))

    n_holdout = max(1, int(len(all_cats) * holdout_frac))
    holdout_cats = list(rng.choice(all_cats, size=n_holdout, replace=False))
    train_cats = [c for c in all_cats if c not in holdout_cats]

    print(f"Categorias de teste (nunca vistas no treino): {holdout_cats}\n")

    results = []
    for l in layers:
        X, y, cats, prompts = load_layer(npz_path, l)

        cross = eval_cross_category(X, y, cats, train_cats, holdout_cats)
        # Match the in-domain training size to the cross-category one. Without
        # this the in-domain probe trains on 80% of everything while the
        # cross-category probe sees only the training categories, and the gap
        # mixes domain shift with how much data each one got.
        in_domain = eval_grouped_split(X, y, prompts, seed=seed, n_train=cross["n_train"])
        ungrouped = eval_random_split(X, y)

        results.append(
            {
                "layer": l,
                "auc_in_domain": round(in_domain["auc"], 4),
                "auc_cross_domain": round(cross["auc"], 4),
                "gap": round(in_domain["auc"] - cross["auc"], 4),
                "auc_in_domain_ungrouped": round(ungrouped["auc"], 4),
                "leak_inflation": round(ungrouped["auc"] - in_domain["auc"], 4),
                "n_train": cross["n_train"],
                "n_test_in_domain": in_domain["n_test"],
                "n_test_cross": cross["n_test"],
            }
        )

    return {
        "holdout_categories": holdout_cats,
        "train_categories": train_cats,
        "seed": seed,
        "per_layer": results,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="data/generalization_report.json")
    args = ap.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    layers = args.layers or sorted(
        int(k.split("_")[1]) for k in data.files if k.startswith("layer_")
    )

    report = run_sweep(args.activations, layers)

    print(f"{'layer':>6} | {'in-domain':>10} | {'cross-domain':>13} | {'gap':>7} | "
          f"{'ungrouped':>10} | {'leak':>7}")
    for r in report["per_layer"]:
        print(f"{r['layer']:>6} | {r['auc_in_domain']:>10.3f} | "
              f"{r['auc_cross_domain']:>13.3f} | {r['gap']:>+7.3f} | "
              f"{r['auc_in_domain_ungrouped']:>10.3f} | {r['leak_inflation']:>+7.3f}")
    print("\n'ungrouped' and 'leak' are diagnostics, not results: they show what the "
          "row-wise\nsplit added on top of the grouped in-domain score.")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatório completo em {args.out}")
