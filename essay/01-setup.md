# Chapter 1 — Setup
### What we are going to build, and the book we are going to build it on

---

## The book

Megillat Esther is 176 verses long. It is the source text of the holiday of Purim, and it is the only book in the Tanakh in which the name of God does not appear on the surface even once.

Two pieces of cultural background are worth having before we begin.

The heroine's name, **אסתר**, has long been read as deriving from the root **סתר** — *to hide*. This is the conventional Sages' reading and it is taught with the book itself. The Megillah is a book in which the central religious presence is hidden from view and in which the heroine herself carries a name meaning "hidden."

The holiday around the book has an unusual halakhic obligation, recorded in the Talmud (Megillah 7b):

> *חייב איניש לבסומי בפוריא עד דלא ידע בין ארור המן לברוך מרדכי*
> *A person is obligated to drink on Purim until he does not know the difference between "cursed is Haman" and "blessed is Mordecai."*

This is the source of the modern term **עדלאידע** — *ad-lo-yada*, "until he does not know" — used today both for the obligation and for the Purim parade tradition.

Two ideas, then, that the culture around this book already treats as central: **hiding** (one name absent from the surface; the heroine named after concealment) and **not-being-able-to-tell-apart** (the holiday's obligation organized around the collapse of the Haman/Mordecai distinction). We will not be making any larger argument about them in this repo. They are just there in the background, and they happen to describe two operations a computer can do mechanically.

---

## What we will build

A small AI, every step from scratch, trained on the 176 verses and nothing else. No outside Hebrew. No prior knowledge. No Tanakh, no Talmud, no dictionary. The model's entire universe is one short book.

Then we will use it for one task: take a verse that mentions Haman or Mordecai, hide the name with a `[MASK]` token, and ask the model to predict — from the surrounding words alone — which of the two characters was originally there.

The classifier scores around **0.84** on this task. Majority-class baseline is 0.54; random is 0.50. The 30-point margin over majority is what the model has learned from the surroundings. The 16% it gets wrong are verses where the author deliberately wrote in mirror form across characters — we will look at those specifically in the error chapter.

The classifier is the apparatus. Building it requires the standard NLP pipeline, and at each stage of the pipeline the model — trained on nothing but this book — ends up with something to say about how the book is written.

---

## And the commercial LLMs?

Most readers will know — at least as users — at least one of ChatGPT, Gemini, or Claude. Those systems are built on the same underlying primitives we will use here: tokenization, embeddings, prediction from context. The differences are scale and architecture (transformers instead of Word2Vec, trillions of tokens of training data instead of our five thousand). Each chapter ahead ends with a short note on how the stage in our small pipeline relates to the corresponding piece of a commercial LLM. The pedagogical claim is that watching these mechanisms on a corpus where every choice matters is the fastest way to understand what they actually do at scale.

---

## What's ahead

| # | Stage | What we build | What the model has to tell us at this stage |
|---|---|---|---|
| 2 | Tokenization | Split the text into reusable bricks | `המלך` is 4.8% of all tokens, twice the next most common |
| 3 | Embedding | Place each token in a 50-dimensional space | The institution is closer to Haman than to Achashverosh on the map |
| 4 | *Interlude A* | Chapter centroids from the same embeddings | The book's chiastic mirrors fall out of the geometry |
| 5 | *Interlude B* | Procrustes-align the two halves of the book | `יקר` moves more between halves than any other word |
| 6 | Cast shifts | Count names by chapter | Each character has a measurable arc; the heroine is absent from the pivot |
| 7 | The classifier | Mask names, train a tiny MLP | 0.84 — beating majority by 30 points |
| 8 | Error analysis | Look at where the classifier fails | The failures cluster on the mirror verses the author wrote in parallel form across characters |

The two **interludes** (Chapters 4 and 5) are short chapters that interrupt the main pipeline. We do not strictly need them for the classifier, but the embedding space we train at Stage 3 also lets us ask two more questions about the book.

A short **appendix** at the end of the repo, for readers who want more technical depth, tests one further hypothesis about the geometry of character substitution and reports a clean negative result. It is optional.

---

**Interactive Walkthrough:** [01_setup.ipynb](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/notebooks/01_setup.ipynb)  
**Code Implementation:** [data.py](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/data.py)
