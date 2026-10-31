"""
Gera um .npz sintético no MESMO formato que extraction.py produziria,
para validar que probe.py e evaluate.py funcionam de ponta a ponta sem
precisar de um modelo real. Não prova nada sobre o fenómeno em si, só
prova que o código não está partido.

Simula 3 layers com separabilidade crescente (layer inicial = ruído puro,
layer do meio = sinal fraco, layer final = sinal forte + algum leakage
de categoria), para verificar que eval_cross_category de facto deteta
overfitting a domínio quando ele existe.

Reproduz também o emparelhamento do dataset real: cada pergunta gera duas
linhas (grounded e confabulada) que partilham um vetor próprio da pergunta,
porque no dataset real a ativação é o último token de "prompt + claim". É isso
que torna o leakage do split por linhas visível aqui, e que permite verificar
que o split agrupado o elimina.
"""

import numpy as np

rng = np.random.RandomState(0)

N = 1580
DIM = 128
CATEGORIES = ["Misconceptions", "Law", "Health", "Sociology", "Economics",
              "Fiction", "Conspiracies", "Paranormal", "Stereotypes", "History"]

# N/2 perguntas, cada uma com uma linha label=1 e uma label=0
N_QUESTIONS = N // 2
question_ids = np.repeat(np.arange(N_QUESTIONS), 2)
prompts = np.array([f"question_{q:04d}" for q in question_ids])
labels = np.tile([1, 0], N_QUESTIONS)
categories = np.repeat(rng.choice(CATEGORIES, size=N_QUESTIONS), 2)

# Vetor próprio de cada pergunta, partilhado pelas duas linhas do par. Sem isto
# o split por linhas não teria nada para memorizar e o leakage não apareceria.
question_vec = rng.randn(N_QUESTIONS, DIM)
question_vec /= np.linalg.norm(question_vec, axis=1, keepdims=True)

cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}
cat_ids = np.array([cat_to_id[c] for c in categories])

# vetor de categoria (ortogonal ao vetor de "grounded"), simula confound de domínio
cat_vec = rng.randn(len(CATEGORIES), DIM)
cat_vec /= np.linalg.norm(cat_vec, axis=1, keepdims=True)

ground_vec = rng.randn(DIM)
ground_vec /= np.linalg.norm(ground_vec)

def make_layer(signal_strength: float, category_leakage: float, noise: float = 1.0,
               question_strength: float = 1.2):
    X = rng.randn(N, DIM) * noise
    X += signal_strength * np.outer(labels * 2 - 1, ground_vec)
    X += category_leakage * cat_vec[cat_ids]
    X += question_strength * question_vec[question_ids]
    return X

layer_low = make_layer(signal_strength=0.1, category_leakage=0.0)     # layer inicial: quase só ruído
layer_mid = make_layer(signal_strength=0.9, category_leakage=0.3)     # sinal real + algum leakage
layer_high = make_layer(signal_strength=1.4, category_leakage=1.6)    # sinal forte MAS confundido com categoria

np.savez_compressed(
    "data/activations_synthetic.npz",
    layer_4=layer_low,
    layer_16=layer_mid,
    layer_28=layer_high,
    labels=labels,
    categories=categories,
    prompts=prompts,
)
print("data/activations_synthetic.npz criado:", N, "exemplos,", DIM, "dims, 3 layers")
