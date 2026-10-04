# Appendix — What We Couldn't Predict
### A clean negative result on the replacement-parallel hypothesis (optional; for readers who want a more technical extension)

---

This is an appendix, not part of the main pipeline. The eight main chapters give a complete account of building and using the classifier. This appendix tests one further hypothesis suggested by the embedding work in Chapters 3–5 — using a more technical move (vector arithmetic on character-difference directions) and a more demanding robustness setup (10 seeds plus bootstrap resampling). It is included for readers who want to see how a clean negative result is established on a corpus this small.

The hypothesis the appendix tests is reasonable. The Megillah is built on structural inversions. Two of them are central: Vashti is replaced by Esther; Haman is replaced by Mordecai. Both are the same *kind* of move — a fall, then a substitution. If the model has captured the geometry of the book, then "being replaced" might exist as a single direction in vector space.

It doesn't.

---

## The hypothesis

In Word2Vec, semantic relations often live as **directions**. The classic example outside this corpus is `vec(king) − vec(man) + vec(woman) ≈ vec(queen)` — the "royalty without gender" direction is a stable axis you can subtract and add. If a relation is *the same* in many places, it shows up as a vector.

The two inversions in the Megillah are at least narratively analogous. If they are also *vectorially* analogous — if the book's `ונהפוך הוא` is a single move applied in two different places — then:

$$ \text{vec}(\text{אסתר}) - \text{vec}(\text{ושתי}) \;\approx\; \text{vec}(\text{מרדכי}) - \text{vec}(\text{המן}) $$

That is the hypothesis. Compute both difference vectors. Compute the cosine between them. If it is high, the book's two great replacements share a single geometric direction. Run it across multiple seeds, then bootstrap it on 80% resamples of the verses, and you get an honest answer.

---

## The result

Across 10 random seeds of the Skip-gram model:

| Quantity | mean | std | range |
|---|---:|---:|---|
| `cos( (אסתר − ושתי), (מרדכי − המן) )` | **0.160** | 0.045 | [0.064, 0.229] |
| Null: cosines between random character-pair differences | 0.212 | 0.205 | [−0.180, 0.590] |

The hypothesis's signal (0.160) is *lower* than the mean of the null distribution (0.212), and well inside the null's spread. The two replacement directions are no more parallel than any other pair of character-difference vectors you could pick at random.

Bootstrap confirms this with a different cut. Resample 80% of the verses 20 times, retrain, recompute:

- Mean cosine across bootstrap samples: **0.162 ± 0.093**
- Fraction of bootstrap samples with cos > 0.3: **5%**
- Fraction with cos > 0: 95% (i.e., the directions are slightly more aligned than antialigned, but the alignment is not meaningfully large)

The cleanest reading: the book's two great inversions do not share a single direction in this embedding space. The hypothesis fails.

---

## Why it failed

Distributional models learn structural relations as directions only when they see the relation repeated many times across the corpus. The classic `king − man + woman = queen` works because the corpora it's trained on contain thousands of male/female noun pairs that share the same context patterns. With enough examples, the model can isolate "gender" as a stable axis.

The Megillah has two replacement pairs. That is not enough examples for a direction to crystallize. There is no statistical pressure on the model to align `אסתר − ושתי` with `מרדכי − המן`, because there is no third or fourth pair to triangulate against. The geometry has the freedom to put each pair wherever the verse-level co-occurrences put it.

Worse, the two pairs are not lexically analogous. Vashti and Esther share many context tokens (queen, banquet, royal house, beauty). Haman and Mordecai share a different set (gate, king's servants, ring, decree). The *contents* of each replacement live in different verses with different vocabulary. The narrative analogy between them — "the rejected one is replaced by the chosen one" — is a *meta-pattern* about the book that does not surface anywhere in the local co-occurrence statistics.

This is the same blind spot that made the chiasm finding possible. The chiasm works because chapters 3 and 8 are *built from the same words*; the inversion between them lives at a different level. The replacement parallel fails for the same structural reason: the analogy lives at the level of *what the book is doing*, not at the level of *which words appear with which*.

---

## What the model can't see

This chapter is itself about what the model can't see, so the limit is the whole point. The technique can detect lexical parallelism (same vocabulary in two places, with the same proportions). It cannot detect semantic analogy across pairs with very few examples and disjoint vocabulary. The book of Esther has many examples of the first kind (the chiasm) and very few of the second (the replacement parallel). The technique succeeds and fails in exactly the places this distinction predicts.

There is also a methodological limit worth stating. With a corpus this small (~5,000 tokens), even a clean *positive* result like the chiasm has to be verified across seeds before it can be reported. A clean *negative* result has to be verified twice as carefully — across seeds, and across bootstrap resamples — because a small sample is the most likely place to mistake noise for signal *or* to mistake signal for noise. The numbers in this chapter survived both checks. The hypothesis is rejected on real evidence, not on a single run that happened to come up flat.

---

## And the commercial LLMs?

Reporting a negative result is unusual for an AI demonstration. Commercial LLMs are trained — through a process called *Reinforcement Learning from Human Feedback*, or RLHF — to be agreeable and useful. They will often produce confident-sounding "yes" answers to leading questions that should produce "no." Asked something like *"isn't `(אסתר − ושתי)` parallel to `(מרדכי − המן)` in the book's geometry?"* a commercial model might well produce a plausible affirmation rather than a measurement — agreement is the response RLHF rewards. Our small system has no such bias because it has no incentive to please anyone. When the hypothesis fails, the measurement says so and there is nowhere for the result to hide. The willingness to land on a clean negative is not a property of model size. It is a property of whether the system is built to *measure* or to *converse*.

## Closing

The chiasm exists, and the model finds it (Chapter 4). The honor word flips, and the model points to it (Chapter 5). Characters live in different vocabulary worlds, and the classifier learns to distinguish them 72% of the time (Chapters 7–8). The replacement parallel is plausible literary intuition, and the model rejects it (this chapter). Each is a real outcome of the same calibrated technique. The repo is a record of which intuitions held up to measurement and which did not.

---

**Code:** [`src/esther_nlp/replacement_parallel.py`](../src/esther_nlp/replacement_parallel.py)
**Reproduce:** `make replacement-parallel` (about three minutes — 10 seeds plus 20-sample bootstrap)

---

**Code Implementation:** [plots.py (bootstrap plot)](../src/esther_nlp/plots.py)
