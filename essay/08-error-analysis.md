# Chapter 8 — Error Analysis
### Where the classifier fails, and why

---

The classifier from the previous chapter scores 0.84 on the masked-character task. The remaining 16% — the 10 verses it gets wrong out of 61 — are not randomly distributed. This chapter looks at them.

## The errors by direction

| True class → Predicted | count |
|---|---:|
| Haman → Mordecai | 4 |
| Mordecai → Haman | 6 |
| Total errors | 10 of 61 |

Roughly symmetric. The model has a mild bias toward predicting Haman when context is ambiguous (6 vs 4) but no class is collapsing.

## The patterns in the errors

The 10 misclassified verses share two visible patterns.

**(1) Errors are longer than correct predictions.** Mean non-`[MASK]` token count for correct predictions: 18.7. For errors: 24.6 — about 30% longer. The intuition that "more context should help" is wrong here. Longer verses tend to contain more shared court vocabulary — `המלך`, `אשר`, `את`, decree formulas — none of which discriminates between the characters. More tokens, in this corpus, is more *noise* relative to signal.

**(2) Errors cluster on the verses the author wrote in mirror form across characters.** This is the deeper pattern. Reading the 10 misclassified verses, almost all of them fall into one of two compositional categories: **chiastic-mirror verses** (verses about Mordecai whose vocabulary matches a corresponding verse about Haman, or vice versa) and **chapter-6 inversion verses** (where Haman's vocabulary literally migrates to Mordecai inside the ironic-honoring scene). Both categories are exactly the structures the rest of the repo has been measuring.

## Two concrete examples

### Example A — Mordecai's counter-decree (Esther 8:9)

> *"ויקראו ספרי המלך בעת ההיא בחדש השלישי הוא חדש סיון... ויכתב ככל אשר צוה מרדכי אל היהודים... אשר מהדו ועד כוש שבע ועשרים ומאה מדינה..."*

True character: Mordecai. Classifier's prediction: **Haman**, with 74% confidence.

This verse is Mordecai writing the counter-decree that saves the Jews. It is also the chapter-8 mirror of Esther 3:12, which describes *Haman* dictating his original decree of destruction. Both verses use almost the same vocabulary: *"the king's scribes were called... it was written according to all that [X] commanded... to all the provinces... in their own writing and language."* This is precisely the chapter-3 ↔ chapter-8 parallelism that Chapter 4's chiastic-mirror analysis measured at cosine 0.97 between the chapter centroids. The vocabulary parallelism is so tight that the classifier reads the verse as Haman's — *because the author wrote it to look like Haman's*. The literary structure that gave Chapter 4 its strongest finding is the structure that fools the classifier here.

### Example B — Mordecai's introduction (Esther 2:5)

> *"איש יהודי היה בשושן הבירה ושמו מרדכי בן יאיר בן שמעי בן קיש איש ימיני"*

True character: Mordecai. Classifier's prediction: **Haman**, with 95% confidence — its single most confident wrong answer.

The verse is Mordecai's introduction-and-genealogy. The classifier is fooled because Haman has a structurally identical introduction-and-genealogy verse in Esther 3:1: *"the king promoted Haman the son of Hamedatha the Agagite..."* The two characters share an introduction *template*: the verbal frame `שמו [name] בן [father] בן [grandfather] ה[ethnicity]`. The masked `[MASK]` of the Mordecai verse is surrounded by the standard introduction-template tokens, and the classifier — which has learned that this template more often signals Haman in the corpus — confidently predicts Haman.

## What the 16% are about

Both examples make the same point. **The verses the classifier fails on are the verses the Megillah's author wrote in deliberate mirror or parallel form across the two characters.** Mordecai's decree mirrors Haman's decree. Mordecai sending letters mirrors Haman sending letters. Mordecai's introduction mirrors Haman's introduction. Two of the misclassified verses are from the chapter-6 honoring scene, where Haman's honor-vocabulary is being transferred to Mordecai inside the central irony of the book.

This is the same compositional fact Chapter 4 measured at the chapter level (the chiastic mirrors that emerge from chapter centroids), and the same blind spot Chapter 9 (the appendix) shows the model cannot overcome at the vector-direction level. **The classifier's confident wrong answers are the literary technique working as the author designed it.** The author wrote these verses to be lexically interchangeable across characters. The classifier confirms the design by being unable to tell them apart.

That gives the 16% a precise meaning. It is not the failure rate of a 4-unit MLP. It is the proportion of the corpus where the author chose, on purpose, to write Mordecai's verses in Haman's verbal clothing — and Haman's verses in Mordecai's. The classifier locates them.

## What the model can't see

The model has no notion of plot, scene, or authorial intent. It cannot reason that "this is Mordecai's counter-decree mirroring Haman's original decree." It only sees that the surface tokens are the same. The literary fact that one verse inverts the other in *meaning* is invisible to a system that processes tokens by co-occurrence. This is the same blind spot we will name explicitly in the optional appendix.

There is also a more general limit. The classifier reports a probability for each prediction. Probabilities above some threshold (say 0.6) are reasonably accurate; probabilities near 0.5 are essentially uncertain. We do not implement a selective-classification variant in this repo, but the structure exists in the predictions — a production use would have to choose a confidence threshold and an "abstain" option for verses where the model has no signal.

## And the commercial LLMs? — Hallucinations

The failure mode we just looked at — confident prediction with no real basis in the input — is the same mechanism that produces **hallucinations** in ChatGPT, Gemini, and Claude. The architecture is responsible in both cases.

Our tiny classifier outputs a probability for each of two classes and picks the higher one. It will *always* output a class. On the Mordecai-introduction verse (Esther 2:5), the surviving tokens after the mask look almost identical to Haman's own introduction verse — and the classifier produces "Haman" at 95% confidence. There is no "I don't know" option in the architecture. The model produces an answer because producing an answer is what the architecture does.

Commercial LLMs have the same structural limit at industrial scale. When you ask ChatGPT a question about a paper that does not exist, and it invents a plausible-sounding citation with a fabricated DOI, it is not malfunctioning. It is doing exactly what our 95%-confidence classifier does on Mordecai's introduction verse: producing the most probable answer its inputs nudge it toward, where the inputs do not constrain a correct answer. Both systems are next-token-prediction machines, and a next-token-prediction machine always emits a next token. Hallucination is not a bug. It is the architecture working as designed.

Three things worth taking from this. **First**, hallucinations are *predictable*. Just as we can identify in advance which verses of the Megillah are likely to be misclassified (the ones the author wrote in mirror form across characters), the commercial models hallucinate more on inputs where they have no real grounding — niche facts, recent events, rare names, contexts that look like training-data patterns but aren't. **Second**, no amount of scale fully fixes this. GPT-4 hallucinates less than GPT-2 because it has more training data and better-tuned probabilities, not because the architecture has learned to abstain. **Third**, the apparent confidence of an LLM output is structural noise, not evidence of correctness. Our classifier reports 0.95 confidence on a wrong answer about Mordecai's lineage. A commercial LLM is the same machine, scaled up. When it sounds certain about a citation that does not exist, the certainty is the architecture; the citation is the same kind of fill-in-a-guess we just watched on a 4-unit MLP.

The reason this matters: a user who reads only chapters 1 through 7 might think modern AI is a *better* version of what we built. It is not, in this respect. It is a much bigger version of what we built, and the same failure mode — confident prediction where no signal exists — is present in both, at the same place in the architecture. The 16% in this chapter is the small visible version of an effect that runs everywhere in the systems people use every day.

---

The main pipeline ends here. An optional [appendix](appendix-replacement-parallel.md) tests one further hypothesis with a more technical method and reports a clean negative result. It is for readers who want to see what an honest "no" looks like on this corpus. Skip it freely if you do not.

→ [Back to the README](../README.md)

---

**Interactive Walkthrough:** [08_error_analysis.ipynb](../notebooks/08_error_analysis.ipynb)  
**Code Implementation:** [classifier.py (evaluation)](../src/esther_nlp/classifier.py)
