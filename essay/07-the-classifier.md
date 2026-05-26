# Chapter 7 — The Classifier
### Predicting the masked character from context

---

We now have everything we need. We have a tokenizer that gives us bricks, an embedding space that gives each brick a 50-dimensional position, and a sense of which character appears where in the book. The task this chapter performs is the one the whole pipeline has been building toward.

## The task

For every verse that mentions exactly one of Haman or Mordecai (61 verses qualify — 8 further verses mention both and are excluded as ambiguous), do three things:

1. Replace the character's name with a `[MASK]` token. Since both `המן` and `מרדכי` are atomic BPE bricks (Stage 2), masking is a single-brick replacement.
2. Build a fixed-length feature vector for the verse from the embeddings of the surviving (unmasked) tokens.
3. Train a small classifier to predict, from the feature vector alone, which character was originally there.

This is structurally identical to BERT's masked-language-modeling objective, restricted to a 2-class label set. It is the most "modern AI" piece in this repo.

## The features

The verse-level feature vector concatenates four things from the surviving tokens' embeddings:

- the **mean** of the embeddings (50-dim)
- the **maximum** along each dimension (50-dim)
- the **minimum** along each dimension (50-dim)
- the **verse length** (1-dim)

That gives 151 features per verse. The concat-pooling is standard practice and roughly preserves both the "average mood" of the verse (mean) and any extreme features (max, min).

## The model

A tiny MLP: 151 input features → 4 hidden units (ReLU + dropout) → 2 output classes. About 620 trainable parameters total.

The architecture is intentionally small. With ~50 training examples per fold, anything larger memorizes the training set and tells us nothing about whether the *context* carries the signal. The point of the experiment is the question — *is the surrounding vocabulary enough?* — not the model's capacity.

Evaluation is 5-fold stratified cross-validation, with per-fold standardization to prevent leakage. Every verse is in the test set exactly once across the folds.

## The result

**Mean test accuracy: 0.84 ± 0.10** across folds.

For comparison: random guessing is 0.50; always predicting the more-common class (Mordecai) is 0.54. The 30-point margin over majority is what the model has learned from the surroundings.

Per-class accuracy is balanced:

| True class | Predicted Haman | Predicted Mordecai | Per-class accuracy |
|---|---:|---:|---:|
| Haman | 24 | 4 | 86% |
| Mordecai | 6 | 27 | 82% |

No class collapse. The model distinguishes both directions at roughly the same rate.

## What the 30-point margin means

The classifier is detecting, indirectly, what Chapter 6 showed directly: each character lives in a different vocabulary neighborhood. Haman's verses are statistically rich in tokens like `חמה` (wrath), `יקר` (honor), `הכסף` (the silver), `ויספר` (and he told). Mordecai's verses are rich in `שער` (the gate), `ישב` (sitting), `התך` (Hathach, his messenger), `ויקרע` (and he tore). When those tokens survive the masking, the model has enough to read which character was hidden.

The 16% the model gets wrong are not random. We'll look at them in the next chapter.

## What the model can't see

The model is reading the corpus at the level of token co-occurrence, not at the level of meaning. It does not understand that Haman is a villain or that Mordecai is loyal. It only knows that the bricks `חמה`, `הכסף`, `יקר` co-occur with `המן` in the training folds and that the bricks `שער`, `ישב`, `התך` co-occur with `מרדכי`. The 0.72 accuracy is a measurement of how much character information is recoverable from token co-occurrence — not a claim about understanding.

The sample size also matters. 61 examples in 50 dimensions is small. The ±0.05 standard deviation across folds is real; numbers reported here are honest within that band, no more precise.

## And the commercial LLMs?

ChatGPT, Gemini, or Claude would score near 100% on this exact task — but not for the reason it might appear. They would not be reading the surrounding words better than our classifier; they would be **recognizing** the verse. Megillat Esther is part of every commercial LLM's training data, often many times over (the Tanakh itself, translations, commentaries, study materials). A commercial model identifying the masked character is more like a person who has memorized the book reciting the next word than like a person reasoning from context. The 0.84 we measure is the honest answer to a different question: *how much of the character's identity is recoverable from the surrounding words when no memorization is available?* Commercial LLMs cannot answer this version of the question, because they have no way to "forget" what they have already read. A small system trained from scratch on the book is the only setup in which the question is testable.

→ [Stage 8 — Error Analysis](08-error-analysis.md)

---

**Interactive Walkthrough:** [07_classifier.ipynb](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/notebooks/07_classifier.ipynb)  
**Code Implementation:** [classifier.py](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/classifier.py)
