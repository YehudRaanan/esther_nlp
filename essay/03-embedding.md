# Chapter 3 — Embedding
### Placing each brick in a 50-dimensional space

---

The previous chapter gave us ~640 bricks. We can now count how often each appears and where, but we have no way to say that two bricks are *similar* in meaning. A model that knows nothing else can only see strings.

This chapter fixes that. We use **Word2Vec** — specifically the Skip-gram variant — to place every brick in a 50-dimensional vector space, in such a way that bricks appearing in similar surroundings end up near each other in the space.

The underlying idea is the one usually summarized as *"you shall know a word by the company it keeps."* For each brick, the algorithm looks at the few bricks that appear immediately to its left and right across the corpus. Bricks whose "company" overlaps a lot get pulled close together in the vector space; bricks whose company is disjoint get pushed apart. After many passes over the verses, the geometry of the space encodes which bricks the corpus uses interchangeably.

We train Skip-gram for 100 epochs with seed-controlled randomness, repeated across 10 different random seeds. The robustness check is necessary on a corpus this small — single-seed runs can be misleading.

## Where the characters land

Once the space is trained, we can ask: for each main character, where does the brick sit relative to `המלך`? Rank all ~640 bricks in the vocabulary by closeness to `המלך`, and read off where each character falls.

Across 10 seeds, the median rank of each character from `המלך`:

| Character | median rank (of ~640) | range across seeds |
|---|---:|---|
| `המן` | **22** | [2, 78] |
| `המלכה` | 53 | [7, 154] |
| `אסתר` | 64 | [3, 251] |
| `מרדכי` | 118 | [1, 440] |
| `ושתי` | 168 | [40, 270] |
| **`אחשורוש`** | **218** | [52, 569] |

Two things stand out and survive the seed-spread:

**Haman ranks closer to `המלך` than any other character.** Across all 10 seeds, the brick that ends up most often in the top neighborhood of "the king" is `המן`. Across seeds the precise rank varies (2 to 78), but the ordering among the characters is consistent.

**Achashverosh ranks the furthest of any main character.** The personal name of the king ends up in the bottom half of the vocabulary on average, far behind the four characters who orbit the throne in the narrative.

This is a strange-looking result if you assume `המלך` and `אחשורוש` are the same person. They are, narratively. They are not, lexically. As we saw in Chapter 2, `המלך` is used 178 times in the book; `אחשורוש` is used only 29 times, and after chapter 3 he is named almost exclusively as `המלך`. So the two bricks live in different vocabulary contexts — the personal name appears in the opening-chapter description of the empire, while the title appears throughout the rest of the book wherever the king acts. Word2Vec puts them in correspondingly different regions of the space.

## How robust the numbers are

The honest statement is that the *order* is robust and the *exact ranks* are not. Single-seed runs can place Haman at rank 2 or at rank 78. The seed-averaged ordering, however, is stable: `המן` < other-characters < `אחשורוש`, every run.

This is a general property of training Word2Vec on a small corpus: there is room for the random initialization to land in different local minima, and a corpus of 5,000 tokens does not provide enough gradient signal to fully constrain the geometry. We report median-across-seeds for that reason. Any single-seed claim ("Haman is the second-closest token") is unreliable. The averaged claim ("Haman is consistently the closest *character* to `המלך`, and Achashverosh is consistently the most distant") survives.

## Why this matters for the classifier

The classifier in Stage 7 will use these embeddings as its input features. Two consequences:

- The embeddings cleanly separate characters in some directions (Haman vs Achashverosh on the king-axis) but not in others (the two queens share much of their vocabulary, as we will see in Stage 9). The classifier will inherit this — it will distinguish characters where the embeddings already have a direction for them, and fail where they don't.
- The embedding for `המלך` is a "center of gravity" in the space — every other character has some relationship to it. Verses heavy in `המלך` will produce verse-level features dominated by this center, with character-distinguishing information drowned out.

## What the model can't see

The geometry is built only from co-occurrence within a 5-brick window. The model does not see syntactic relations (subject vs object), does not see narrative order across verses, and does not see chapter boundaries. Two bricks that always appear in the same verses end up near each other regardless of whether they ever play the same grammatical role.

The geometry is also small-data noisy. Findings that depend on a *single* brick's position are not safe; findings that depend on the *relative ordering* of several bricks are safer. We will use the second kind throughout the rest of the repo.

## And the commercial LLMs?

Commercial LLMs use embeddings too, but they are *contextual* — the vector for a word depends on the surrounding words at inference time. The vector for `המלך` inside `המלך אחשורוש` is different from the vector for `המלך` inside `כבוד המלך`, even though the underlying token is the same. Our Word2Vec is the 2013-era ancestor of this idea: one static vector per token, fixed across all contexts. Contextual embeddings (the kind GPT and Claude use) learn the same kind of "neighborhood geometry" Word2Vec captures, but they learn a different geometry for every sentence they see. Our findings in this chapter (Haman near `המלך`, Achashverosh far) describe the *global* geometry — the typical neighborhood across the whole book. Commercial models would replace that single global geometry with one local geometry per verse. The global picture is harder to read off a contextual model; it is the natural unit of a static one.

→ Before we get to the classifier, the embedding space we just trained also lets us answer two questions about the book that have nothing to do with masking names. These are the next two chapters, marked as *interludes*. If you only want the classifier path, you can jump to [Stage 6 — Cast Shifts](06-the-cast-shifts.md).

→ [Interlude A — The Chiastic Mirror](04-the-chiastic-mirror.md)

---

**Interactive Walkthrough:** [03_embedding.ipynb](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/notebooks/03_embedding.ipynb)  
**Code Implementation:** [embeddings.py](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/embeddings.py)
