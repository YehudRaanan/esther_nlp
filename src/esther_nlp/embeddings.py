"""Word2Vec embeddings wrapper and Orthogonal Procrustes alignment."""

from typing import List, Set, Tuple

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes


def train_word2vec(
    tokenized_corpus: List[List[str]],
    sg: int = 1,
    epochs: int = 100,
    seed: int = 42,
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 2
) -> Word2Vec:
    """Trains a Word2Vec model on the tokenized corpus.

    Args:
        tokenized_corpus: Nested list of BPE string tokens.
        sg: Architecture (1 = Skip-gram, 0 = CBOW).
        epochs: Number of training epochs.
        seed: Random state seed to pin determinism.
        vector_size: Dimension size of word vectors (default 50).
        window: Sliding window size (default 5).
        min_count: Minimum token occurrence to embed (default 2).

    Returns:
        A trained Gensim Word2Vec model instance.
    """
    return Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,  # Pinned to 1 worker to ensure exact seed reproducibility
        epochs=epochs,
        seed=seed
    )


def align_spaces_procrustes(
    model1: Word2Vec,
    model2: Word2Vec,
    counts_total: dict,
    min_corpus_freq: int = 3
) -> Tuple[List[Tuple[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """Aligns vector space of model1 to model2 using Orthogonal Procrustes alignment.

    Measures semantic shifts (1 - cosine) between the aligned representations.

    Args:
        model1: Source embedding model (e.g. trained on chapters 1–5).
        model2: Target embedding model (e.g. trained on chapters 6–10).
        counts_total: Total dictionary of word frequencies in overall corpus.
        min_corpus_freq: Threshold to filter out rare tokens.

    Returns:
        A tuple containing:
        - List of (word, shift_distance) pairs sorted by shift descending.
        - Source normalized vectors (A).
        - Source aligned/rotated vectors (A_rotated).
        - Target normalized vectors (B).
    """
    # Identify shared vocabulary between both models
    shared_vocab: Set[str] = set(model1.wv.key_to_index) & set(model2.wv.key_to_index)
    
    # Filter out subword fragments (e.g. beginning with ##) and rare tokens
    shared_words: List[str] = [
        word for word in shared_vocab 
        if not word.startswith("##") and counts_total.get(word, 0) >= min_corpus_freq
    ]
    
    # Build L2-normalized matrix layers
    A = np.stack([
        model1.wv[w] / (np.linalg.norm(model1.wv[w]) + 1e-12) 
        for w in shared_words
    ])
    B = np.stack([
        model2.wv[w] / (np.linalg.norm(model2.wv[w]) + 1e-12) 
        for w in shared_words
    ])
    
    # Run Orthogonal Procrustes: find orthogonal matrix R to minimize ||A @ R - B||
    R, _ = orthogonal_procrustes(A, B)
    A_rotated = A @ R
    
    # Calculate word shifts (1 - cosine_similarity of aligned positions)
    shifts = []
    for idx, word in enumerate(shared_words):
        a_rot = A_rotated[idx]
        b_vec = B[idx]
        cos_sim = float(a_rot @ b_vec / (np.linalg.norm(a_rot) * np.linalg.norm(b_vec) + 1e-12))
        shifts.append((word, 1.0 - cos_sim))
        
    # Sort by shift descending (largest shift = most mobile)
    shifts.sort(key=lambda x: -x[1])
    
    return shifts, A, A_rotated, B
