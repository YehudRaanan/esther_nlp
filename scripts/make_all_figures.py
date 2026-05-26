"""Regenerates all 12 publication-ready figures for the Megillah and the Machine.

Runs the complete end-to-end NLP pipeline and passes metrics into the plots visualizer.
"""

import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from esther_nlp.data import load_chapters_and_verses, get_verse_location_map
from esther_nlp.tokenizer import EstherBpeTokenizer, compute_vocabulary_counts
from esther_nlp.embeddings import train_word2vec, align_spaces_procrustes
from esther_nlp.classifier import (
    build_classifier_examples,
    featurize_sentence,
    run_cross_validation
)
from esther_nlp.analysis import (
    get_radial_ranks,
    compute_chapter_centroids,
    compute_centroid_similarity_matrix,
    ablate_epochs_convergence
)
import esther_nlp.plots as plots

# Resolve output folders
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------
print("1. Loading Megillat Esther dataset...")
chapters, all_verses = load_chapters_and_verses()

print("2. Tokenizing and counting vocabulary...")
tokenizer = EstherBpeTokenizer(vocab_size=1000)
tokenizer.train(all_verses)
tokenized_verses = [tokenizer.encode_to_tokens(v) for v in all_verses]
tokenized_by_ch = [[tokenizer.encode_to_tokens(v) for v in ch] for ch in chapters]
counts_total = compute_vocabulary_counts(tokenized_verses)
total_tokens = sum(counts_total.values())

print("3. Training deterministic Skip-gram Word2Vec embeddings (10 seeds)...")
SEEDS = [42, 1, 7, 13, 21, 99, 100, 314, 1729, 2025]
sg_models = [train_word2vec(tokenized_verses, sg=1, epochs=100, seed=s) for s in SEEDS]
base_model = sg_models[0]  # Pinned base seed (42)

print("4. Re-running the PyTorch MLP classifier (5-fold stratified CV)...")
examples = build_classifier_examples(all_verses, tokenizer)
X = np.stack([featurize_sentence(m, base_model, dim=50) for m, _, _, _ in examples])
y = np.array([lbl for _, lbl, _, _ in examples], dtype=np.int64)

# Re-run CV using classifier module
mean_acc, fold_accs, preds, confs = run_cross_validation(X, y, n_splits=5, random_state=42, epochs=200)
clf_std = float(np.std(fold_accs))
majority_baseline = max((y == c).sum() / len(y) for c in range(2))

# -----------------------------------------------------------------------------
# Figure 1: Pipeline flow
# -----------------------------------------------------------------------------
print("   -> Generating Figure 1: NLP Pipeline Overview...")
plots.fig_01_pipeline(FIG_DIR / "fig_01_pipeline.png")

# -----------------------------------------------------------------------------
# Figure 2: Top tokens bar chart
# -----------------------------------------------------------------------------
print("   -> Generating Figure 2: Top BPE Tokens frequency...")
top_15 = counts_total.most_common(15)
tokens_15 = [t for t, _ in top_15]
counts_15 = [c for _, c in top_15]
plots.fig_02_top_tokens(tokens_15, counts_15, total_tokens, FIG_DIR / "fig_02_top_tokens.png")

# -----------------------------------------------------------------------------
# Figure 3: Radial target map
# -----------------------------------------------------------------------------
print("   -> Generating Figure 3: Radial character distances from המלך...")
sorted_chars = ["המן", "המלכה", "אסתר", "מרדכי", "ושתי", "אחשורוש"]
per_char = {c: [] for c in sorted_chars}
vocab_size = len(base_model.wv.key_to_index) - 1  # Excluding המלך

for model in sg_models:
    ranks = get_radial_ranks(model, "המלך", sorted_chars)
    for c in sorted_chars:
        r, _ = ranks.get(c, (-1, 0.0))
        per_char[c].append(r)
        
medians = {c: int(np.median(per_char[c])) for c in sorted_chars}
sorted_by_med = sorted(sorted_chars, key=lambda c: medians[c])
plots.fig_03_radial_map(sorted_by_med, medians, per_char, vocab_size + 1, FIG_DIR / "fig_03_radial_map.png")

# -----------------------------------------------------------------------------
# Figure 4: Chapter similarity arcs
# -----------------------------------------------------------------------------
print("   -> Generating Figure 4: Chapter-centroid similarities...")
sims_across_seeds = []
for model in sg_models[:5]:
    centroids = compute_chapter_centroids(model, tokenized_by_ch)
    sims_across_seeds.append(compute_centroid_similarity_matrix(centroids))
mean_sim_matrix = np.mean(sims_across_seeds, axis=0)
plots.fig_04_arc_diagram(mean_sim_matrix, FIG_DIR / "fig_04_arc_diagram.png")

# -----------------------------------------------------------------------------
# Figure 5a & 5b: Procrustes semantic moves and neighbor tag clouds
# -----------------------------------------------------------------------------
print("   -> Generating Figure 5: Procrustes alignment shifts and neighbor clouds...")
first_half_v  = [v for ch in chapters[:5] for v in ch]
second_half_v = [v for ch in chapters[5:] for v in ch]

def get_tokens(texts):
    tk = EstherBpeTokenizer(vocab_size=1000)
    tk.train(texts)
    return [tk.encode_to_tokens(v) for v in texts]

m1 = train_word2vec(get_tokens(first_half_v),  sg=1, epochs=100, seed=42, vector_size=50)
m2 = train_word2vec(get_tokens(second_half_v), sg=1, epochs=100, seed=42, vector_size=50)

shifts, _, _, _ = align_spaces_procrustes(m1, m2, counts_total, min_corpus_freq=3)

# 5a: Top 10 movers horizontal bars
top_movers_10 = shifts[:10]
movers_tokens = [w for w, _ in top_movers_10]
movers_values = [v for _, v in top_movers_10]
plots.fig_05a_top_movers(movers_tokens, movers_values, FIG_DIR / "fig_05a_top_movers.png")

# 5b: Neighbor tag clouds of יקר
n1 = m1.wv.most_similar("יקר", topn=10) if "יקר" in m1.wv else []
n2 = m2.wv.most_similar("יקר", topn=10) if "יקר" in m2.wv else []
plots.fig_05b_neighbor_clouds(n1, n2, FIG_DIR / "fig_05b_neighbor_clouds.png")

# -----------------------------------------------------------------------------
# Figure 6: Streamgraph character streams
# -----------------------------------------------------------------------------
print("   -> Generating Figure 6: Character streamgraph rivers...")
char_groups = {
    "המלך":     ["המלך"],
    "אחשורוש":  ["אחשורוש", "אחשורש"],
    "ושתי":     ["ושתי"],
    "אסתר":     ["אסתר", "לאסתר"],
    "מרדכי":    ["מרדכי", "ומרדכי", "למרדכי", "במרדכי"],
    "המן":      ["המן", "והמן", "להמן", "מהמן"],
    "היהודים":  ["היהודים", "ביהודים", "ליהודים"],
}
density = np.zeros((len(char_groups), 10), dtype=float)
char_names = list(char_groups.keys())

for ci, ch_tokens in enumerate(tokenized_by_ch):
    flat_tokens = [token for verse in ch_tokens for token in verse]
    for ri, (_, group) in enumerate(char_groups.items()):
        density[ri, ci] = sum(flat_tokens.count(g) for g in group)
plots.fig_06_streamgraph(density, char_names, FIG_DIR / "fig_06_streamgraph.png")

# -----------------------------------------------------------------------------
# Figure 6b: 3D embedding ball
# -----------------------------------------------------------------------------
print("   -> Generating Figure 6b: 3D character vector ball...")
ball_chars = ["המלך", "המן", "מרדכי", "אסתר", "אחשורוש"]
ball_vectors = np.stack([base_model.wv[c] for c in ball_chars])

# Project 50D to 3D PCA, then L2-normalize to unit sphere coordinates
vn = ball_vectors / (np.linalg.norm(ball_vectors, axis=1, keepdims=True) + 1e-12)
p_pca = PCA(n_components=3).fit(vn)
pos_3d = p_pca.transform(vn)
pos_3d = pos_3d / (np.linalg.norm(pos_3d, axis=1, keepdims=True) + 1e-12)
plots.fig_06b_embedding_ball(pos_3d, ball_chars, FIG_DIR / "fig_06b_embedding_ball.png")

# -----------------------------------------------------------------------------
# Figure 7: Accuracy bar comparison
# -----------------------------------------------------------------------------
print("   -> Generating Figure 7: Classifier accuracy vs baseline comparison...")
plots.fig_07_accuracy(majority_baseline, mean_acc, clf_std, FIG_DIR / "fig_07_accuracy.png")

# -----------------------------------------------------------------------------
# Figure 8a: Sankey confusion flow
# -----------------------------------------------------------------------------
print("   -> Generating Figure 8a: Sankey confusion flows...")
confusion = np.zeros((2, 2), dtype=int)
for truth, prediction in zip(y, preds):
    confusion[truth, prediction] += 1
plots.fig_08a_sankey(confusion, FIG_DIR / "fig_08a_sankey.png")

# -----------------------------------------------------------------------------
# Figure 8b: Confidence scatter
# -----------------------------------------------------------------------------
print("   -> Generating Figure 8b: Softmax confidence scatters...")
plots.fig_08b_hallucinations(confs, preds == y, FIG_DIR / "fig_08b_hallucinations.png")

# -----------------------------------------------------------------------------
# Figure 10b: Yakar dual 3D vector spheres
# -----------------------------------------------------------------------------
print("   -> Generating Figure 10b: Yakar side-by-side 3D spheres...")
n1_all = [w for w, _ in base_model.wv.most_similar("יקר", topn=5)]
n1_readable = [w for w in n1_all if len(w) > 1 and not w.startswith("##")][:3]

n2_all = [w for w, _ in train_word2vec(get_tokens(second_half_v), sg=1, epochs=100, seed=42, vector_size=50).wv.most_similar("יקר", topn=10)]
n2_readable = [w for w in n2_all if len(w) > 1 and not w.startswith("##")][:3]

plots.fig_10b_yakar_two_worlds(m1, m2, n1_readable, n2_readable, FIG_DIR / "fig_10b_yakar_two_worlds.png")

# -----------------------------------------------------------------------------
# Figure Appendix: Bootstrap histograms
# -----------------------------------------------------------------------------
print("   -> Generating Figure Appendix: Bootstrap alignment histograms...")
def get_vec(m, t):
    return m.wv[t].copy() if t in m.wv else None

def get_cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

# Generate null samples
null_pairs = [
    ("אסתר", "המן",   "מרדכי", "ושתי"),
    ("ושתי", "המן",   "אסתר",  "מרדכי"),
    ("אחשורוש", "ושתי", "אחשורוש", "המן"),
    ("המלך", "המן",   "המלך",  "מרדכי"),
]
null_samples = []
for m in sg_models:
    for a, b, c, d in null_pairs:
        va, vb, vc, vd = get_vec(m, a), get_vec(m, b), get_vec(m, c), get_vec(m, d)
        if all(x is not None for x in (va, vb, vc, vd)):
            null_samples.append(get_cos(va - vb, vc - vd))

# Run bootstrap resampling (80% resample, 20 times)
boot_samples = []
rand_gen = random.Random(2026)
for i in range(20):
    sample_idx = rand_gen.sample(range(len(all_verses)), int(0.8 * len(all_verses)))
    sub_verses = [all_verses[k] for k in sample_idx]
    
    sub_tk = EstherBpeTokenizer(vocab_size=1000)
    sub_tk.train(sub_verses)
    sub_tokenized = [sub_tk.encode_to_tokens(v) for v in sub_verses]
    
    m_boot = train_word2vec(sub_tokenized, sg=1, epochs=100, seed=42 + i, vector_size=50)
    e, v, mo, h = get_vec(m_boot, "אסתר"), get_vec(m_boot, "ושתי"), get_vec(m_boot, "מרדכי"), get_vec(m_boot, "המן")
    if any(x is None for x in (e, v, mo, h)):
        continue
    boot_samples.append(get_cos(e - v, mo - h))

plots.fig_appendix_bootstrap(np.array(null_samples), np.array(boot_samples), FIG_DIR / "fig_appendix_bootstrap.png")

print(f"\nSUCCESS: All 12 figures successfully regenerated and saved to: {FIG_DIR.resolve()}")
