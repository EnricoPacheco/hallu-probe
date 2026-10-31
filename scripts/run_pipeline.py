"""
Orquestra o pipeline completo. Corre no teu ambiente local (com GPU e
acesso a huggingface.co), não neste sandbox.

Passo 1 e 4 (dataset, probe, evaluate) já correm em CPU e sem internet
extra além do que já foi buscado. Passo 2 (extração) e 5 (steering)
precisam do modelo real.

    python scripts/run_pipeline.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, -1])
    ap.add_argument("--limit", type=int, default=None,
                     help="limitar nº de exemplos, útil para um primeiro teste rápido")
    args = ap.parse_args()

    py = sys.executable

    run([py, "src/dataset.py", "--raw", "data/truthfulqa.csv", "--out", "data/labeled_claims.csv"])

    extract_cmd = [
        py, "src/extraction.py",
        "--model", args.model,
        "--data", "data/labeled_claims.csv",
        "--out", "data/activations.npz",
        "--layers", *map(str, args.layers),
    ]
    if args.limit:
        extract_cmd += ["--limit", str(args.limit)]
    run(extract_cmd)

    run([py, "src/probe.py", "--activations", "data/activations.npz"])
    run([py, "src/evaluate.py", "--activations", "data/activations.npz"])

    print("\nPipeline completo. Ver data/generalization_report.json para o resultado "
          "que decide se vale a pena avançar para steering.py.")
