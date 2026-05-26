"""Analytical metrics and quantitative calculations for Megillat Esther."""

from typing import Dict, List, Tuple

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

from esther_nlp.embeddings import train_word2vec


def get_radial_ranks(
    model: Word2Vec,
    anchor: str,
    targets: List[str]
) -> Dict[str, Tuple[int, float]]:
    """Calculates the similarity ranks and cosine values of target words from an anchor.

    Ranks targets out of the entire vocabulary based on cosine similarity desc.

    Args:
        model: Trained Word2Vec model.
        anchor: Reference word (e.g., 'המלך').
        targets: List of characters/words to evaluate (e.g., ['המן', 'מרדכי']).

    Returns:
        Dict mapping target -> (rank_1_based, cosine_similarity).
    """
    if anchor not in model.wv:
        raise ValueError(f"Anchor word '{anchor}' not in model vocabulary.")
        
    # Compute similarity to all other vocabulary words
    sims = []
    for word in model.wv.key_to_index:
        if word == anchor:
            continue
        sims.append((word, float(model.wv.similarity(anchor, word))))
        
    # Sort by cosine similarity descending
    sims.sort(key=lambda x: -x[1])
    
    # Map target words to ranks
    vocab_ranks = {}
    for idx, (word, cos_val) in enumerate(sims, start=1):
        if word in targets:
            vocab_ranks[word] = (idx, cos_val)
            
    # Add placeholders for targets not in vocabulary
    for target in targets:
        if target not in vocab_ranks:
            vocab_ranks[target] = (-1, 0.0)
            
    return vocab_ranks


def compute_chapter_centroids(
    model: Word2Vec,
    tokenized_by_chapter: List[List[List[str]]]
) -> np.ndarray:
    """Computes an L2-normalized centroid vector for each chapter.

    Centroid is the mean of all vocabulary word vectors in the chapter's verses.

    Args:
        model: Trained Word2Vec model.
        tokenized_by_chapter: Nested BPE tokens list [chapters -> verses -> tokens].

    Returns:
        A 2D numpy array of shape (num_chapters, vector_dim).
    """
    centroids = []
    for ch_tokens in tokenized_by_chapter:
        # Collect and L2-normalize individual token vectors
        vectors = []
        for verse in ch_tokens:
            for token in verse:
                if token in model.wv:
                    vec = model.wv[token]
                    norm = np.linalg.norm(vec) + 1e-12
                    vectors.append(vec / norm)
                    
        # Compute chapter mean centroid
        if vectors:
            ch_mean = np.mean(vectors, axis=0)
            ch_mean = ch_mean / (np.linalg.norm(ch_mean) + 1e-12)
            centroids.append(ch_mean)
        else:
            centroids.append(np.zeros(model.vector_size))
            
    return np.stack(centroids)


def compute_centroid_similarity_matrix(C: np.ndarray) -> np.ndarray:
    """Computes the cosine similarity matrix between chapter centroids.

    Args:
        C: Matrix of shape (num_chapters, vector_dim).

    Returns:
        Symmetric matrix of shape (num_chapters, num_chapters).
    """
    nrm = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return (C / nrm) @ (C / nrm).T


def get_loneliest_verses(
    tokenized_verses: List[List[str]],
    models: List[Word2Vec],
    top_n: int = 10
) -> List[Tuple[int, float]]:
    """Calculates the 'loneliest' (most semantically distinctive) verses.

    Computes verse vectors using multi-seed averaged embeddings, then finds
    verses with the lowest average cosine similarity to all other verses.

    Args:
        tokenized_verses: List of BPE string token lists.
        models: List of trained Word2Vec models (e.g. trained with different seeds).
        top_n: Number of loneliest verses to return.

    Returns:
        List of tuples: (global_verse_index, mean_cosine_similarity).
    """
    # Extract shared vocabulary across all models
    shared_vocab = set(models[0].wv.key_to_index)
    for model in models[1:]:
        shared_vocab &= set(model.wv.key_to_index)
        
    # Pre-compute L2-normalized vector averaged across seeds
    def get_seed_avg_vector(token: str) -> np.ndarray:
        vectors = []
        for m in models:
            v = m.wv[token]
            vectors.append(v / (np.linalg.norm(v) + 1e-12))
        return np.mean(vectors, axis=0)
        
    avg_embeddings = {token: get_seed_avg_vector(token) for token in shared_vocab}
    
    # Featurize each verse
    verse_vectors = []
    for tokens in tokenized_verses:
        valid_vecs = [avg_embeddings[t] for t in tokens if t in avg_embeddings]
        if valid_vecs:
            v_vec = np.mean(valid_vecs, axis=0)
            v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-12)
            verse_vectors.append(v_vec)
        else:
            verse_vectors.append(np.zeros(models[0].vector_size))
            
    V = np.stack(verse_vectors)
    
    # Compute similarity matrix and nan-out diagonal (verse similarity to itself)
    sim_matrix = V @ V.T
    np.fill_diagonal(sim_matrix, np.nan)
    
    mean_sims = np.nanmean(sim_matrix, axis=1)
    loneliest_indices = np.argsort(mean_sims)[:top_n]
    
    return [(idx, float(mean_sims[idx])) for idx in loneliest_indices]


def get_first_appearance_timeline(
    tokenized_verses: List[List[str]],
    char_groups: Dict[str, List[str]],
    verse_loc_map: Dict[int, Tuple[int, int]]
) -> Dict[str, dict]:
    """Computes the verse and chapter indices for each character's first and last appearances.

    Args:
        tokenized_verses: List of BPE string token lists.
        char_groups: Dictionary mapping character labels to equivalent tokens.
        verse_loc_map: Global verse index -> (chapter_1_based, verse_in_ch_1_based) map.

    Returns:
        Dict mapping character name -> timeline metadata dictionary.
    """
    timeline = {}
    for name, tokens_group in char_groups.items():
        appearances = []
        for v_idx, tokens in enumerate(tokenized_verses):
            if any(tok in tokens for tok in tokens_group):
                appearances.append(v_idx)
                
        if not appearances:
            timeline[name] = {
                "first_verse": -1, "first_chapter": -1,
                "last_verse": -1, "last_chapter": -1
            }
            continue
            
        first, last = appearances[0], appearances[-1]
        ch_first, _ = verse_loc_map[first]
        ch_last, _ = verse_loc_map[last]
        
        timeline[name] = {
            "first_verse": first,
            "first_chapter": ch_first,
            "last_verse": last,
            "last_chapter": ch_last
        }
        
    return timeline


def ablate_epochs_convergence(
    tokenized_verses: List[List[str]],
    epochs_list: List[int],
    seed: int = 42,
    vector_size: int = 50
) -> List[dict]:
    """Ablates Word2Vec convergence: tracks Haman's rank from המלך across epochs.

    Args:
        tokenized_verses: List of BPE string token lists.
        epochs_list: List of epoch numbers to test.
        seed: Random state seed.
        vector_size: Word vector size.

    Returns:
        List of dictionaries containing epoch results.
    """
    results = []
    for ep in epochs_list:
        m = train_word2vec(tokenized_verses, sg=1, epochs=ep, seed=seed, vector_size=vector_size)
        if "המלך" not in m.wv or "המן" not in m.wv:
            results.append({
                "epochs": ep, "rank": -1, "cosine": 0.0, "top_5": []
            })
            continue
            
        # Get targets ranks
        ranks = get_radial_ranks(m, "המלך", ["המן"])
        haman_rank, haman_cos = ranks.get("המן", (-1, 0.0))
        
        # Get top-5 neighbors of המלך
        sims = sorted(
            ((t, float(m.wv.similarity("המלך", t))) for t in m.wv.key_to_index if t != "המלך"),
            key=lambda x: -x[1]
        )
        top_5 = sims[:5]
        
        results.append({
            "epochs": ep,
            "rank": haman_rank,
            "cosine": haman_cos,
            "top_5": top_5
        })
        
    return results
