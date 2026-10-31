"""
Fase 2 do projeto, só faz sentido depois de confirmar que o probe generaliza
(ver evaluate.py). Usa a direção grounded->confabulado (mean-diff, calculada
em probe.mean_diff_direction) e soma/subtrai essa direção nas ativações
durante a geração, via forward hook numa layer específica.

Isto é a parte que ainda não tem resposta certa: empurrar a ativação na
direção "grounded" reduz alucinação de facto, ou só muda o estilo da
resposta sem mudar o conteúdo factual? É o que dá pra medir comparando
a taxa de acerto no TruthfulQA com e sem steering.

Uso:
    python src/steering.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --activations data/activations.npz \
        --layer 16 --alpha 4.0 \
        --prompt "What happens if you crack your knuckles a lot?"
"""

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe import load_layer, mean_diff_direction


class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.direction.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        else:
            return output + self.alpha * self.direction.to(output.dtype).to(output.device)


def generate_with_steering(model_name: str, prompt: str, direction: np.ndarray,
                             layer: int, alpha: float, device: str, max_new_tokens: int = 80):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    direction_t = torch.tensor(direction / (np.linalg.norm(direction) + 1e-8))

    inputs = tok(prompt, return_tensors="pt").to(device)

    def run(steer: bool):
        handle = None
        if steer:
            target_layer = model.model.layers[layer]
            hook = SteeringHook(direction_t, alpha)
            handle = target_layer.register_forward_hook(hook)
        try:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        finally:
            if handle is not None:
                handle.remove()
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    baseline = run(steer=False)
    steered = run(steer=True)
    return baseline, steered


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations", default="data/activations.npz")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    X, y, _ = load_layer(args.activations, args.layer)
    direction = mean_diff_direction(X, y)

    baseline, steered = generate_with_steering(
        args.model, args.prompt, direction, args.layer, args.alpha, args.device
    )

    print("=== baseline ===")
    print(baseline)
    print("\n=== com steering (direção grounded, alpha=%.1f) ===" % args.alpha)
    print(steered)
