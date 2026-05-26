# Chapter 2 — Tokenization
### Splitting the 176 verses into reusable bricks

---

To do anything mathematical with the text, we first need to turn it into discrete countable units. Word boundaries help — the verses are space-separated — but we want something smaller and more flexible than whole words. We use a technique called **Byte Pair Encoding (BPE)**.

The idea is simple. Start with each individual Hebrew letter as its own brick. Then repeatedly look for the most frequent pair of adjacent bricks across the whole corpus, merge them into a single new brick, and add it to the vocabulary. Repeat. The result is a dictionary of bricks ranging from single letters (rare) to whole words (common), built entirely from the text itself with no Hebrew prior. The algorithm is given no grammar, no dictionary, no list of stopwords. It just counts.

We train BPE on the 176 verses with a vocabulary cap of 1,000.

## What came out

The trained tokenizer ends up using ~640 bricks (the rest are unfilled — BPE is allowed up to 1,000 but only uses the merges it needs). All seven main characters become single, indivisible bricks:

| Token | In vocab as single brick? |
|---|---|
| `המלך` | yes |
| `המלכה` | yes |
| `אחשורוש` | yes |
| `אסתר` | yes |
| `מרדכי` | yes |
| `המן` | yes |
| `ושתי` | yes |

This is convenient for what we will do in Stage 7. To hide a character's name from the classifier, we will replace exactly one brick.

## The dominant brick

One token sticks out. The top 10 BPE tokens by frequency:

| Token | count | % of corpus |
|---|---:|---:|
| **`המלך`** | **178** | **4.8%** |
| `אשר` | 91 | 2.4% |
| `את` | 80 | 2.2% |
| `על` | 56 | 1.5% |
| `אל` | 52 | 1.4% |
| `אסתר` | 43 | 1.2% |
| `מרדכי` | 42 | 1.1% |
| `המן` | 39 | 1.0% |
| `ואת` | 38 | 1.0% |
| `כי` | 36 | 1.0% |

`המלך` is **1.96× more frequent** than the next most common token, which is the relativizer `אשר`. For comparison: the English word *"the"* — the most frequent word in English running text — typically makes up around 5–7% of all tokens. `המלך` in Megillat Esther is appearing at roughly the rate "the" appears in English. It is a content word with the frequency profile of a function word.

The Sages of the Talmud noticed this and taught that every unnamed `המלך` in the Megillah should be read as referring to the King of Kings. We are not making any larger claim about that reading. The empirical fact is reportable on its own: in this corpus the word `המלך` dominates the frequency distribution to a degree no other content word approaches.

## Why this matters for the classifier

Three concrete consequences for our classifier:

- **Masking will be a single-brick operation.** Both `המן` and `מרדכי` exist as atomic tokens. The mask replaces exactly one brick per verse.
- **The classifier's input will be dominated by `המלך`.** Whichever verses we mask, the surrounding tokens will be heavy with `המלך`. This is unavoidable on this corpus — there are too few alternatives.
- **Verses dominated by `המלך` will be the hardest to classify.** When the surviving context is mostly the dominant brick and a few function words, neither character has distinctive signal. We will see this play out in Stage 8.

## What tokenization can't see

Once we have bricks, the model has no access to the individual letters inside each brick. Letter-level patterns — acrostics, gematria, root-letter analyses — are invisible from this point forward. We are committing to the brick level for everything downstream.

The choice of vocabulary size matters. A larger vocabulary would have kept more whole words as single bricks; a smaller one would have broken more words into morphemes. We chose 1,000 because the corpus is too small to support a much larger vocabulary meaningfully and we wanted the main character names as atomic units for the classifier task. Different choices would shift specific numbers downstream, but the qualitative findings in later chapters survive across a reasonable range of vocab sizes.

## And the commercial LLMs?

ChatGPT, Gemini, and Claude all begin with tokenization too. GPT-4 uses a BPE tokenizer with about 100,000 tokens in its vocabulary, trained on terabytes of multilingual text. Claude uses a related BPE variant; Gemini uses SentencePiece. None of these tokenizers is trained primarily on Hebrew. The practical consequence is that for Hebrew text, commercial tokenizers tend to *fragment* — a Hebrew word that we keep as a single brick (like `המלך`) may be broken into three or four sub-pieces by a general-purpose tokenizer, because the algorithm never saw enough Hebrew during training to merge it. The "dominant brick" observation we just made about `המלך` would not survive that fragmentation. We can see the structure clearly here partly *because* we trained on Hebrew exclusively.

→ [Stage 3 — Embedding](03-embedding.md)

---

**Interactive Walkthrough:** [02_tokenization.ipynb](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/notebooks/02_tokenization.ipynb)  
**Code Implementation:** [tokenizer.py](file:///g:/האחסון%20שלי/learning/Nebius/LLM_Architecture/Ex2/repo-draft/src/esther_nlp/tokenizer.py)
