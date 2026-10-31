"""
Extrai as ativações internas (hidden states) de um modelo de peso aberto
para cada par (prompt, claim) do dataset rotulado.

Ideia: o texto completo é "prompt + claim". Passamos pelo modelo com
output_hidden_states=True e guardamos o hidden state do último token,
em várias camadas. É essa representação que o probe vai usar para prever
se a claim é grounded ou confabulada, sem olhar para o texto em si.

Corre local, com GPU. Precisa de acesso a huggingface.co para baixar os
pesos do modelo (bloqueado neste sandbox).

Uso:
    python src/extraction.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data data/labeled_claims.csv \
        --out data/activations.npz \
        --layers 8 16 24 -1
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def extract_activations(model_name: str, df: pd.DataFrame, layers: list[int], device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, output_hidden_states=True, low_cpu_mem_usage=True
    ).to(device)

    n_layers = model.config.num_hidden_layers + 1  # +1 = embedding layer
    resolved_layers = [l if l >= 0 else n_layers + l for l in layers]

    acts_by_layer = {l: [] for l in resolved_layers}
    labels, categories, prompts = [], [], []

    for _, row in df.iterrows():
        text = f"{row['prompt']} {row['claim']}"
        inputs = tok(text, return_tensors="pt").to(device)

        out = model(**inputs)
        hidden_states = out.hidden_states  # tuple: (n_layers+1, batch, seq, dim)

        for l in resolved_layers:
            last_token_vec = hidden_states[l][0, -1, :].float().cpu().numpy()
            acts_by_layer[l].append(last_token_vec)

        labels.append(int(row["label"]))
        categories.append(row["category"])
        # The prompt is the grouping key. Each TruthfulQA question yields one
        # grounded and one confabulated claim, and the activation is the last
        # token of "prompt + claim", so the two share the prompt's
        # representation. Without this column a random split puts one of the
        # pair in train and the other in test and the probe can read the
        # question instead of the grounding.
        prompts.append(row["prompt"])

    save_dict = {f"layer_{l}": np.stack(v) for l, v in acts_by_layer.items()}
    save_dict["labels"] = np.array(labels)
    save_dict["categories"] = np.array(categories)
    save_dict["prompts"] = np.array(prompts)
    return save_dict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ex: meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default="data/labeled_claims.csv")
    ap.add_argument("--out", default="data/activations.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, -1])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="limitar nº de linhas p/ teste rápido")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    print(f"Extraindo ativações de {len(df)} exemplos com {args.model} em {args.device}...")
    result = extract_activations(args.model, df, args.layers, args.device)
    np.savez_compressed(args.out, **result)
    print(f"Guardado em {args.out}")
