# Zeresh — a tiny, fully sober MLP
### *A Transparent and Simple NLP Sandbox on 167 Verses*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Tests Passing](https://img.shields.io/badge/tests-9%20passed-green.svg)](tests/)

An end-to-end, highly rigorous Data Science pipeline trained from scratch on the **167 consonantal verses** of Megillat Esther — and nothing else. No pretrained embeddings, no transfer learning, and no prior Hebrew lexicon or dictionary. 

The primary objective of this project is to explore how standard machine learning architectures (BPE tokenization, Word2Vec semantic spaces, and Multi-Layer Perceptrons) behave when trained on a corpus small enough that every parameters decision carries massive visual and narrative consequences.

Our core model is named **Zeresh (זרש)** — a play on Haman's wife and the Talmudic Purim obligation of *Ad-lo-yada* (to drink until one cannot distinguish between "blessed is Mordecai" and "cursed is Haman"). Since Zeresh (our machine) is completely sober, she attempts to solve the masked character classification task: hiding a character's name in a verse and predicting, from surrounding words alone, whether the verse was originally about Haman or Mordecai. 

**Zeresh scores exactly 83.7% accuracy.** The 16.3% of verses she misclassifies are not random errors; they are the exact verses the author wrote as deliberate **chiastic mirror templates** across both characters — representing the exact structural counterpart of hallucinations in commercial Large Language Models (LLMs).

---

## 🎓 Demystifying LLM "Magic" for Everyone

One of the greatest challenges of modern artificial intelligence is that Large Language Models (LLMs) operate as massive, multi-billion-parameter "black boxes," making their underlying mechanics feel like magic. 

**Zeresh** is intentionally designed to solve this by serving as an **intuitive, highly accessible NLP sandbox**. It is specifically crafted for **non-professional users (requiring only a basic, high-level familiarity with the story of Megillat Esther)** to easily demystify and understand the core components that power today's AI systems:

1. **Tokenization (How Machines See Text)**: Instead of complex grammatical rules, we show how Byte Pair Encoding (BPE) automatically splits words into functional sub-word fragments based purely on character frequencies. 
2. **Word Embeddings (How Machines Define Meaning)**: We demonstrate how machines represent words as physical coordinates in a geometric space. A non-technical user can visually inspect how words that share similar contexts naturally cluster together (e.g., seeing **המן** and **המלך** sitting adjacent on a 3D sphere because they co-occur in royal court contexts).
3. **Neural Networks & "Hallucinations" (How Machines Predict and Err)**: By training a tiny PyTorch Multi-Layer Perceptron (MLP) to predict characters, we expose the exact math of neural predictions. More importantly, we show that AI "errors" are not arbitrary bugs, but the logical result of structural symmetries in the input training data—giving an intuitive, clean model of why LLMs "hallucinate."

By scaling the entire LLM pipeline down to a tiny, closed-world corpus of exactly 167 verses, we make every mathematical transformation completely transparent, visual, and trace-level visible for anyone wanting to understand how AI truly works.

---

## 🗺️ Codebase Map & Structure

The repository has been refactored to align with industry-grade, best-practice open-source Data Science layouts:

```
esther_nlp/
├── data/
│   └── esther_dataset.txt              # The 167 consonantal verses of Megillat Esther
├── fonts/
│   └── Rubik-Variable.ttf              # Bundled open-license Rubik font for cross-platform plots
├── src/
│   └── esther_nlp/
│       ├── __init__.py                 # Package setup
│       ├── data.py                     # Verse & chapter loading and location maps
│       ├── tokenizer.py                # BPE tokenizer training and serialization wrappers
│       ├── embeddings.py               # Word2Vec Skip-gram/CBOW wrapper and Orthogonal Procrustes
│       ├── classifier.py               # PyTorch ZereshMLP and stratified 5-fold cross-validation
│       ├── analysis.py                 # Chapter centroids, radial ranks, and lonely verses
│       └── plots.py                    # Matplotlib visualizers (RTL / BiDi reshaping support)
├── notebooks/                          # 8 highly-polished Jupyter Notebook walkthroughs
│   ├── 01_setup.ipynb                  # Data loading and pipeline preview
│   ├── 02_tokenization.ipynb           # BPE merging and המלך dominance frequency
│   ├── 03_embedding.ipynb              # Semantic spaces and Haman's closeness to the throne
│   ├── 04_chiastic_mirror.ipynb        # Macro centroid symmetries (decrees and banquets)
│   ├── 05_honor_that_flips.ipynb       # Orthogonal Procrustes alignment of the shift word 'יקר'
│   ├── 06_cast_shifts.ipynb            # Dynamic streamgraph rivers of protagonist mentions
│   ├── 07_classifier.ipynb             # Featurization and Zeresh MLP training
│   └── 08_error_analysis.ipynb         # Confident hallucinations and chiastic mirror errors
├── essay/                              # Master long-form prose essay chapters
│   ├── 01-setup.md
│   ├── ...
│   └── appendix-replacement-parallel.md
├── figures/                            # 12 pre-generated publication-ready PNG plots
├── tests/                              # Automated test suite (determinism, leakage, parsing)
│   ├── test_data.py
│   ├── test_tokenizer.py
│   ├── test_embeddings.py
│   ├── test_classifier.py
│   └── test_determinism.py             # Enforces cross-platform seed-pinned numerical determinism
├── results/
│   └── metrics_snapshot.json           # Committed JSON snapshot of every reported value
├── Makefile                            # Standardized developer task runner
├── pyproject.toml                      # Package config and python dependencies management
└── README.md                           # This premium landing page
```

---

## 🚀 Setup & Reproducibility

Determinism is 100% verified via automated tests. Every figure and reported number is completely reproducible.

### Quick Start (Local Setup)

This project manages environment packaging and dependencies with modern tools. You can install in editable mode:

```bash
# Clone the repository
git clone https://github.com/YehudRaanan/esther_nlp.git
cd esther_nlp

# Setup environment and install dependencies
pip install -e .[dev]

# Run the complete test suite (collection takes ~9 seconds)
PYTHONPATH=src pytest
```

### Unified Task Runner Commands

All common project workflows are standardized under a `Makefile`:

*   **`make install`**: Installs the package in editable mode with development dependencies.
*   **`make test`**: Runs the complete automated regression test suite (asserts seed-pinned data and classifier scores).
*   **`make run`**: Executes the entire end-to-end NLP pipeline and exports a fresh JSON snapshot of results to `results/metrics_snapshot.json`.
*   **`make figures`**: Re-generates all 12 publication-ready plots into the `figures/` directory.
*   **`make clean`**: Safely cleans out Python build, cache, and test files.

---

## 📊 Key Research Findings

The pipeline reveals a series of striking structural and semantic signatures in the text of the Megillah:

### 1. The Dominance of `המלך` (Stage 2)
The Byte Pair Encoding algorithm trained with a vocabulary size of 1000 identifies `המלך` (the King) as the single most frequent token, appearing 178 times — representing **4.8%** of the entire running text. `המלך` is **1.96× more frequent** than the next most common token (`אשר`), behaving like a syntactic function word (equivalent to the frequency profile of "the" in English).

### 2. Office Over Person (Stage 3)
On our semantic coordinates map, Haman's vector sits at **rank 22** of closeness to `המלך` (in the top 3% of the entire vocabulary), while the personal name `אחשורוש` resides at **rank 218** (the furthest of all main characters). The geometry cleanly splits the office from the human king, reflecting the narrative takeover of the crown's title after chapter 3.

### 3. Symmetries of Fate (Stage 4 & 5)
Chapter centroid similarities independently uncover the Megillah's two famous chiastic mirrors: **Chapter 3 ↔ Chapter 8** (the two decrees) correlate at **cosine 0.973**, and **Chapter 5 ↔ Chapter 7** (the two banquets) correlate at **cosine 0.967**. Additionally, Orthogonal Procrustes alignment identifies **`יקר` (yakar / honor)** as the single most mobile word in the vocabulary, migrating from passive court-pomp neighbors (`כבוד`, `קטן`) in the first half to irony and active national celebration (`למרדכי`, `אורה`) in the second.

### 4. Sober Errors & LLM Hallucinations (Stage 8)
When Zeresh classifies Haman and Mordecai from context alone, she scores **83.7%**. The 10 misclassified verses represent deliberate mirrors where the author wrote identical surrounding context templates to emphasize the turn of fate. For example:
*   **Verse 8:9** (Mordecai writing the defense decree) is classified as **Haman with 74% confidence** because the author wrote it using the exact grammatical blocks of Haman's decree of destruction in **3:12**.
*   **Verse 2:5** (introducing Mordecai) is classified as **Haman with 95% confidence** because the author introduced both using an identical genealogical template (`ושמו X בן Y בן Z...`).

The model is highly confident and completely wrong. This is the exact structural mechanism behind **hallucinations** in modern transformer-based LLMs: templates constrain a highly probable answer that is entirely ungrounded in facts. Hallucination is not a bug — it is the next-token prediction architecture working as designed.

---

## 🎨 Interactive Walkthroughs (Jupyter Notebooks)

Click the absolute paths below to explore and run the interactive stages of the pipeline:

1.  [**Chapter 1 — Setup**](notebooks/01_setup.ipynb): Parsing the consonantal corpus and previewing the pipeline.
2.  [**Chapter 2 — Tokenization**](notebooks/02_tokenization.ipynb): BPE merges and quantitative analysis of the dominant word `המלך`.
3.  [**Chapter 3 — Embedding**](notebooks/03_embedding.ipynb): Word2Vec training and character distance coordinates.
4.  [**Chapter 4 — Chiastic Mirror**](notebooks/04_chiastic_mirror.ipynb): Macro chapter-centroids symmetries and similarities.
5.  [**Chapter 5 — Honor That Flips**](notebooks/05_honor_that_flips.ipynb): Orthogonal Procrustes alignment of the shift word `יקר`.
6.  [**Chapter 6 — Cast Shifts**](notebooks/06_cast_shifts.ipynb): Protagonist rivers streamgraph.
7.  [**Chapter 7 — Zeresh Classifier**](notebooks/07_classifier.ipynb): Masked-character PyTorch MLP training.
8.  [**Chapter 8 — Error & Hallucinations**](notebooks/08_error_analysis.ipynb): Sages' mirror verses and structural model failures.

---

## 📜 Long-Form Essays

The complete, beautifully detailed narrative chapters of our investigation are compiled under the `essay/` directory. Each chapter features full academic prose, pre-generated figures, and clickable links to the codebase modules:

*   [**Chapter 1 — Setup**](essay/01-setup.md)
*   [**Chapter 2 — Tokenization**](essay/02-tokenization.md)
*   [**Chapter 3 — Embedding**](essay/03-embedding.md)
*   [**Chapter 4 — Chiastic Mirror**](essay/04-the-chiastic-mirror.md)
*   [**Chapter 5 — Honor That Flips**](essay/05-the-honor-that-flips.md)
*   [**Chapter 6 — Cast Shifts**](essay/06-the-cast-shifts.md)
*   [**Chapter 7 — Zeresh Classifier**](essay/07-the-classifier.md)
*   [**Chapter 8 — Error Analysis**](essay/08-error-analysis.md)
*   [**Appendix — What We Couldn't Predict**](essay/appendix-replacement-parallel.md) (Optional multi-seed bootstrap test of the replacement-parallel vector direction)

---

## 🎙️ Hebrew Presentation Narrative

For Hebrew speakers, this research was compiled into a comprehensive presentation summarizing the LLM-sandbox approach to Megillat Esther.
*   **Presentation Summary PDF**: [`זרש - מודל השפה שלא שתה.pdf`](זרש - מודל השפה שלא שתה.pdf) — A complete, detailed outline of the Hebrew narrative, slide concepts, and structural findings.

---

## 🪪 License & Credit

*   **Code**: Licensed under the [MIT License](LICENSE).
*   **Dataset**: Megillat Esther consonantal version (without nikkud or cantillation) is in the public domain. Verse alignments are based on the Mechon-Mamre public domain corpus. See `DATA_LICENSE.md` (or license footnotes) for complete citations.
