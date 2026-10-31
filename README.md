# hallu-probe

Can you tell whether a language model is about to state something it knows, or
something it is making up, by looking at its internal activations rather than at
the text it produces?

The hypothesis is that the distinction already exists as a linear direction in
the hidden states, before the token is emitted. If it does, a linear probe on
those activations should separate grounded claims from confabulated ones, and
the same direction should be usable to steer generation. The question that
decides whether any of this is interesting is not whether a probe fits, but
whether it **generalises to topics it was never trained on**.

## Status

Honest summary, because the distinction matters for reading anything below:

**Runs and is tested here (CPU, no network):** dataset construction, probe
training, the generalisation sweep, and an end-to-end check on synthetic
activations.

**Needs a GPU and access to Hugging Face, not yet run:** activation extraction
from a real model, and therefore every result about actual language models.

**No result about any real model is reported in this repository yet.** The
numbers in the sweep below come from synthetic activations and exist only to
show that the measurement code does what it claims. Reading them as findings
about Llama or Mistral would be wrong.

## Method

The dataset is built from TruthfulQA, which supplies, for each question, a
correct answer and a plausible incorrect one. That pairing is the reason for
choosing it: both claims answer the same question in the same register, so the
only variable left is whether the content is factual. 790 questions give 1580
labelled claims across 37 categories.

For each (prompt, claim) pair, the model is run over the concatenated text with
`output_hidden_states=True` and the hidden state of the final token is stored at
several layers. A logistic regression per layer is then trained to predict
grounded versus confabulated from that vector alone.

Two scores are reported per layer, and the gap between them is the point:

- **in-domain**, split grouped by question, with categories mixing freely. This
  measures separability in the easy case.
- **cross-category**, trained on one set of categories and tested on categories
  held out entirely, with the training set trimmed to the same size as the
  in-domain one.

A small gap is evidence that the direction is a general property of the model. A
large gap says the probe learned the vocabulary of its training topics. Both
outcomes are informative, which is why the project is worth running either way.

## Design decisions

**TruthfulQA rather than a bespoke dataset.** The matched-pair structure
controls for topic and phrasing without any work. Building a dataset first would
have meant also arguing that the pairs were comparable.

**The last token, not the mean over the sequence.** The final position is where
the model has committed to what comes next. This is a parameter in
`extraction.py` and other positions are worth testing; it is a starting choice,
not a conclusion.

**Logistic regression, not an MLP.** If a linear classifier separates the
classes, the direction is interpretable and can be used for steering. An MLP
would hide whatever it found behind a nonlinear boundary and leave nothing to
steer with.

**Splitting by category, not k-fold at random.** A random fold mixes categories
across train and test, so it cannot tell whether the probe learned the
phenomenon or the topic. The whole question is topic transfer, so the split has
to be by topic.

**Splitting in-domain by question, not by row.** Each question contributes two
rows that share the prompt, and the activation is taken from the end of
`prompt + claim`, so the two rows share the prompt's representation. A row-wise
split puts one of the pair in training and the other in test: on this dataset
82.9% of test rows have their prompt present in training, and the probe can
recognise the question rather than judge the claim. The in-domain split is
therefore a `GroupShuffleSplit` on the prompt. `eval_random_split` is kept to
measure what the ungrouped version was adding, and the sweep prints it as a
diagnostic rather than a result.

**Matching the training-set size across the two scores.** The cross-category
probe only sees the training categories, so comparing it against an in-domain
probe trained on 80% of everything would fold a sample-size difference into the
gap. The in-domain training set is trimmed to the cross-category size.

## Running it

```bash
pip install -r requirements.txt

# end-to-end on synthetic activations, no GPU and no model download
python tests/make_synthetic_activations.py
python src/evaluate.py --activations data/activations_synthetic.npz

# the real thing: needs a GPU and huggingface.co
python scripts/run_pipeline.py --model meta-llama/Llama-3.1-8B-Instruct --limit 300
```

`--limit 300` subsamples for a first pass that takes minutes instead of hours.
Drop it for the full dataset.

Steering is phase two and only makes sense once the sweep says the direction
generalises:

```bash
python src/steering.py --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 16 --alpha 4.0 \
    --prompt "What happens if you crack your knuckles a lot?"
```

## Code-validation run

Synthetic activations, three layers built with increasing separability, a
category confound, and the question pairing of the real dataset reproduced so
the grouped split is actually exercised:

| layer | in-domain (grouped) | cross-category | gap | ungrouped | leak |
| --- | --- | --- | --- | --- | --- |
| 4 (noise) | 0.454 | 0.529 | -0.075 | 0.487 | +0.034 |
| 16 (moderate signal) | 0.862 | 0.868 | -0.006 | 0.903 | +0.041 |
| 28 (strong signal, confounded) | 0.975 | 0.974 | +0.000 | 0.980 | +0.005 |

What this does and does not show. It shows that the sweep separates the layers
as designed, that the grouped and ungrouped splits differ, and that nothing is
silently broken. It says nothing about any real model, and the size of the leak
on real activations is unknown until extraction runs.

## Next

1. Run the pipeline on a small open-weight model (Llama-3.2-3B or Mistral-7B)
   with `--limit 300`, to confirm it works end to end on real activations.
2. Read the gap. If it is large at every layer, the project turns into "why
   doesn't this transfer", which is a result worth writing up. If it is small at
   some layers, continue.
3. Test whether a probe trained on one model transfers to a larger model in the
   same family. This is the part nobody has pinned down convincingly.
4. Only then, steering: measure TruthfulQA accuracy with and without it. The
   open question is whether pushing along the direction reduces confabulation or
   only changes the register of the answer.

## Limitations

- TruthfulQA is adversarial by construction, built around questions where models
  are known to fail. A probe that works here may not work on ordinary claims.
- The labels are about the claim, not about what the model believes. A model can
  state a correct fact without any internal representation of knowing it.
- One position, one token, one classifier family. The result is a lower bound on
  what is recoverable, not a measurement of what is there.
- Steering changes activations at inference. Whether a measured accuracy gain
  means less confabulation or a shift in answer style needs a separate
  evaluation that this repository does not have yet.
