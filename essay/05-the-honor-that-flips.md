# Chapter 5 — Interlude B: The Honor That Flips
### Tracking which words change meaning between halves of the book

---

A second question we can ask with the embeddings from Stage 3, again unrelated to our classifier task.

The Megillah's phrase `ונהפוך הוא` ("and it was turned upside down" — 9:1) is the book's own description of the central inversion: Vashti replaced by Esther, Haman's gallows used for Haman, Mordecai inheriting the ring. The inversion is at the event level. But it might also be at the word level — particular words might mean different things in the first half of the book than in the second.

The question for this chapter: does the embedding space we trained earlier let us identify which words change their meaning between halves of the book?

---

## The experiment

Split the book in half. Train Word2Vec independently on chapters 1–5 and on chapters 6–10. Each half now lives in its own 50-dimensional space. Because they were trained separately, the two spaces are in *different coordinate systems* — you cannot directly compare a vector in one to a vector in the other, any more than you can compare two cities' GPS coordinates without agreeing on north.

The fix is a standard linear-algebra move called **Procrustes alignment**. Among all the words that appear in both halves (115 of them), find the rotation of the first space that brings shared words as close as possible to their position in the second space. After this rotation, words whose *meaning didn't change* will sit in nearly the same place in both halves; words whose meaning *did change* will still be far apart.

For each shared word, measure how far apart it ended up after alignment. The top mover, across five different random seeds:

| Token | rank across seeds | mean position |
|---|---|---:|
| **`יקר`** (honor) | [1, 1, 2, 3, 2] | **1.8 / 115** |

Always in the top three. The single most semantically mobile token in the book.

For context, the most stable tokens:

| Token | rank |
|---|---:|
| `אסתר` | 102 / 115 |
| `המלך` | 99 / 115 |

The heroine and the institution sit at almost identical positions in both halves. They are the constants. The book is built around them, and what changes around them is everything else — most of all, `יקר`.

---

## What the word's neighbors say

Looking at the top neighbors of `יקר` in each half makes the shift visible in plain language.

| Half | `יקר`'s top neighbors (Skip-gram top-10) |
|---|---|
| **ch 1–5** | `ועד`, `למגדול`, `קטן`, `ימים`, `כבוד`, `ביום` |
| **ch 6–10** | `נעשה`, `נערי`, `למרדכי`, **`ביקרו`**, `לעשות`, `ויאמרו` |

The first set is the vocabulary of a sweeping decree — "from the great to the small, on a day, with glory." It is `יקר` as the abstract noun a king's edict uses when it tells wives to honor their husbands (1:20).

The second set is the vocabulary of a specific scene — "what was done," "the king's young men," "to Mordecai," "in his honor," "to do." It is `יקר` as the very specific royal investiture performed in chapter 6: the robe, the horse, the procession through the city.

The same Hebrew word. Two different worlds.

---

## The eleven verses

There is nothing hidden about this. Anyone who reads the book carefully can see it — the model just measures it. Every appearance of `יקר` in the Megillah, in chapter order:

- **1:4** — *"showing the wealth of his glorious kingdom and the יקר of his great majesty"*
- **1:20** — *"all wives shall give יקר to their husbands, from great to small"*
- **3:1** — *"the king promoted Haman and raised his throne above all the princes who were with him"* (the related verb `יקר` appears in form)
- **6:3** — *"what יקר and greatness has been done for Mordecai for this?"*
- **6:6** — *"to whom would the king delight to יקר more than to me?"*
- **6:7** — *"the man whom the king delights to יקר…"*
- **6:9** — *"…and let them cry before him: thus shall it be done for the man whom the king delights to יקר"*
- **6:11** — *"…and called before him: thus shall it be done for the man whom the king delights to יקר"*
- **8:16** — *"the Jews had light, and gladness, and joy, and יקר"*

In chapters 1 and 2 the word appears in decree vocabulary — generic court esteem ("wives shall give honor to their husbands"). Six of the eleven occurrences are concentrated in chapter 6, where the word becomes the object of the book's most famous comic scene: Haman, intending to ask the king for permission to hang Mordecai, is preemptively asked by the king what should be done for the man the king delights to honor — and Haman, assuming the king means him, designs a lavish ceremony that he is then forced to perform for Mordecai. By chapter 8 the same word lands on the Jews in the victory verse.

The same word; three different contexts.

---

## What the model can't see

The model has not detected the irony of chapter 6. It has detected that `יקר` lives among different words in the two halves of the book. It cannot tell that the man who designed the ceremony is the man who had to perform it. The content of the inversion is invisible to the technique.

What the technique sees is the surface trace: the word that sits in two different vocabulary neighborhoods. That trace is not the meaning, but it is the place the meaning leaves a footprint. The footprint is detectable; the meaning is not.

## And the commercial LLMs?

Commercial LLMs would not naturally surface this kind of finding. Asked *"which word changes meaning between halves of Esther?"* a commercial model could probably name `יקר` — not by measurement, but because it has seen commentaries on chapter 6 that single out the word. It would not run a Procrustes alignment. The instrument matters. Commercial systems are built for generation and conversation, not for systematic measurement of distributional shift inside a corpus. Small pipelines like this one can do something the big ones are not designed to do: measure a property of *the text itself*, rather than retrieve a property of the *commentary about* the text.

→ Back to the main pipeline. The next chapter switches from embeddings to raw counts.

→ [Stage 6 — The Cast Shifts](06-the-cast-shifts.md)

---

**Interactive Walkthrough:** [05_honor_that_flips.ipynb](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/notebooks/05_honor_that_flips.ipynb)  
**Code Implementation:** [embeddings.py (alignment)](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/embeddings.py) & [analysis.py](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/analysis.py)
