"""
Constrói um dataset rotulado (prompt, claim, label, category) a partir do TruthfulQA.

Cada pergunta do TruthfulQA já vem com uma resposta correta (grounded) e uma
resposta incorreta plausível (confabulada). Isso dá pares perfeitamente
controlados para treinar o probe: mesmo prompt, mesma "forma" de resposta,
só muda se o conteúdo é factual ou inventado.

label = 1 -> claim grounded (factualmente correta)
label = 0 -> claim confabulada (hallucination)
"""

import argparse
import pandas as pd


def build_dataset(raw_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    rows = []
    for _, r in df.iterrows():
        question = str(r["Question"]).strip()
        category = str(r.get("Category", "unknown")).strip()

        best = str(r.get("Best Answer", "")).strip()
        worst = str(r.get("Best Incorrect Answer", "")).strip()

        if best and best.lower() != "nan":
            rows.append(
                {"prompt": question, "claim": best, "label": 1, "category": category}
            )
        if worst and worst.lower() != "nan":
            rows.append(
                {"prompt": question, "claim": worst, "label": 0, "category": category}
            )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["prompt", "claim"]).reset_index(drop=True)
    out.to_csv(out_csv_path, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/truthfulqa.csv")
    ap.add_argument("--out", default="data/labeled_claims.csv")
    args = ap.parse_args()

    df = build_dataset(args.raw, args.out)
    print(f"{len(df)} pares (prompt, claim) escritos em {args.out}")
    print(df["label"].value_counts())
    print(df["category"].value_counts().head(10))
