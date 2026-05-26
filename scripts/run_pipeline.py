"""Executes the complete end-to-end NLP pipeline and exports numerical results."""

import json
from pathlib import Path
import numpy as np

from esther_nlp.data import load_chapters_and_verses, get_verse_location_map
from esther_nlp.tokenizer import EstherBpeTokenizer, compute_vocabulary_counts
from esther_nlp.embeddings import train_word2vec, align_spaces_procrustes
from esther_nlp.classifier import (
    build_classifier_examples,
    featurize_sentence,
    run_cross_validation,
    ID_TO_CLASS
)
from esther_nlp.analysis import (
    get_radial_ranks,
    compute_chapter_centroids,
    compute_centroid_similarity_matrix,
    get_loneliest_verses,
    get_first_appearance_timeline,
    ablate_epochs_convergence
)

# Output snapshot folder
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_SNAPSHOT = RESULTS_DIR / "metrics_snapshot.json"

def main():
    print("=" * 80)
    print("      Megillah and the Machine — End-to-End NLP Execution Pipeline")
    print("=" * 80)
    
    # 1. Data Parsing
    print("\n[1/6] Loading Megillat Esther consonantal corpus...")
    chapters, all_verses = load_chapters_and_verses()
    verse_loc = get_verse_location_map(chapters)
    print(f"      Success: Loaded {len(chapters)} chapters and {len(all_verses)} flat verses.")

    # 2. Tokenization
    print("\n[2/6] Training BPE Tokenizer and encoding texts...")
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    tokenizer.train(all_verses)
    tokenized_verses = [tokenizer.encode_to_tokens(v) for v in all_verses]
    tokenized_by_ch = [[tokenizer.encode_to_tokens(v) for v in ch] for ch in chapters]
    counts_total = compute_vocabulary_counts(tokenized_verses)
    total_tokens = sum(counts_total.values())
    print(f"      Success: Vocabulary compiled. Total unique BPE tokens: {len(counts_total)}")

    # 3. Vector Embeddings
    print("\n[3/6] Modeling Skip-gram Word2Vec embeddings (seed 42)...")
    base_model = train_word2vec(tokenized_verses, sg=1, epochs=100, seed=42, vector_size=50)
    print("      Success: Word vectors trained successfully.")

    # 4. Qualitative Semantic Analysis
    print("\n[4/6] Conducting semantic distance and textual structure analysis...")
    
    # Character rankings
    sorted_chars = ["המן", "המלכה", "אסתר", "מרדכי", "ושתי", "אחשורוש"]
    radial_ranks = get_radial_ranks(base_model, "המלך", sorted_chars)
    print("      - Character semantic closeness toהמלך (the King):")
    for char, (rank, cos) in sorted(radial_ranks.items(), key=lambda x: x[1][0]):
        print(f"        * {char:<10} -> Rank {rank:<3} (cosine {cos:.3f})")
        
    # Chapter similarities (chiastic mirror checking)
    centroids = compute_chapter_centroids(base_model, tokenized_by_ch)
    sim_matrix = compute_centroid_similarity_matrix(centroids)
    print("      - Chapter-pair similarity checking (strongest off-diagonals):")
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append((i, j, float(sim_matrix[i, j])))
    pairs.sort(key=lambda x: -x[2])
    for i, j, val in pairs[:4]:
        print(f"        * Chapter {i+1} <-> Chapter {j+1} -> cosine similarity {val:.3f}")
        
    # Semantic word flip (Procrustes movers)
    print("      - Running Orthogonal Procrustes alignment on first vs second halves...")
    first_half_v  = [v for ch in chapters[:5] for v in ch]
    second_half_v = [v for ch in chapters[5:] for v in ch]
    
    def tokenize_half(texts):
        tk = EstherBpeTokenizer(vocab_size=1000)
        tk.train(texts)
        return [tk.encode_to_tokens(v) for v in texts]
        
    m1 = train_word2vec(tokenize_half(first_half_v),  sg=1, epochs=100, seed=42, vector_size=50)
    m2 = train_word2vec(tokenize_half(second_half_v), sg=1, epochs=100, seed=42, vector_size=50)
    
    shifts, _, _, _ = align_spaces_procrustes(m1, m2, counts_total, min_corpus_freq=3)
    print("      - Top 5 semantic shifts between halves of the book:")
    for word, dist in shifts[:5]:
        print(f"        * {word:<10} -> shift distance {dist:.3f}")
        
    # Loneliest verses
    print("      - Finding the loneliest (most semantically distinctive) verses...")
    # Using multiple seeds for robust calculations
    SEEDS = [42, 1, 7, 21, 99]
    models = [train_word2vec(tokenized_verses, sg=1, epochs=100, seed=s) for s in SEEDS]
    lonely = get_loneliest_verses(tokenized_verses, models, top_n=3)
    for rank_idx, (idx, mean_cos) in enumerate(lonely, start=1):
        ch_idx, v_idx = verse_loc[idx]
        print(f"        * #{rank_idx} distinctive: ch{ch_idx}:{v_idx} (avg cosine {mean_cos:.3f}) -> \"{all_verses[idx][:55]}...\"")
        
    # timelines
    char_groups = {
        "המלך":     ["המלך"],
        "אחשורוש":  ["אחשורוש", "אחשורש"],
        "ושתי":     ["ושתי"],
        "אסתר":     ["אסתר", "לאסתר"],
        "מרדכי":    ["מרדכי", "ומרדכי", "למרדכי", "במרדכי"],
        "המן":      ["המן", "והמן", "להמן", "מהמן"],
    }
    timeline = get_first_appearance_timeline(tokenized_verses, char_groups, verse_loc)

    # 5. Masked Character MLP Classifier
    print("\n[5/6] Building Zeresh MLP masked classification dataset...")
    examples = build_classifier_examples(all_verses, tokenizer)
    X = np.stack([featurize_sentence(m, base_model, dim=50) for m, _, _, _ in examples])
    y = np.array([lbl for _, lbl, _, _ in examples], dtype=np.int64)
    print(f"      Success: Featurized {len(examples)} verses. Input shape: {X.shape}")
    
    print("      Running stratified 5-fold cross-validation...")
    mean_acc, fold_accs, preds, confs = run_cross_validation(X, y, n_splits=5, random_state=42, epochs=200)
    print(f"      Success: Zeresh MLP Classifier Accuracy: {mean_acc * 100:.1f}%")
    print(f"      Individual fold scores: {[f'{a*100:.1f}%' for a in fold_accs]}")

    # 6. Replicability Metrics Snapshots Export
    print("\n[6/6] Exporting reproducible results snapshot to results/...")
    
    # Prep JSON serializable metrics
    snapshot = {
        "dataset": {
            "num_chapters": len(chapters),
            "num_verses": len(all_verses),
            "total_tokens": total_tokens
        },
        "character_closeness_to_king": {
            char: {"rank": r, "cosine": float(c)} for char, (r, c) in radial_ranks.items()
        },
        "chapter_similarities": [
            {"ch1": i+1, "ch2": j+1, "cosine": float(sim_matrix[i, j])} 
            for i in range(10) for j in range(i+1, 10)
        ],
        "top_word_semantic_shifts": [
            {"word": w, "shift": float(d)} for w, d in shifts[:15]
        ],
        "loneliest_verses": [
            {"idx": int(idx), "ch": verse_loc[idx][0], "verse": verse_loc[idx][1], "cosine": float(c)}
            for idx, c in lonely
        ],
        "character_timelines": timeline,
        "classifier_metrics": {
            "num_examples": len(examples),
            "mean_accuracy": float(mean_acc),
            "fold_accuracies": [float(a) for a in fold_accs]
        }
    }
    
    OUT_SNAPSHOT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      Success: Saved deterministic metrics JSON snapshot to: {OUT_SNAPSHOT.resolve()}")
    print("\n" + "=" * 80)
    print("                               Pipeline Run Finished!")
    print("=" * 80)

if __name__ == "__main__":
    main()
