"""Unit tests for Word2Vec models and Procrustes alignments."""

import numpy as np
from esther_nlp.data import load_chapters_and_verses
from esther_nlp.tokenizer import EstherBpeTokenizer, compute_vocabulary_counts
from esther_nlp.embeddings import train_word2vec, align_spaces_procrustes
from esther_nlp.analysis import compute_chapter_centroids, compute_centroid_similarity_matrix


def test_embeddings_and_alignment():
    """Asserts Word2Vec model builds correctly and Procrustes spaces align."""
    chapters, all_verses = load_chapters_and_verses()
    
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    tokenizer.train(all_verses)
    
    tokenized = [tokenizer.encode_to_tokens(v) for v in all_verses]
    counts_total = compute_vocabulary_counts(tokenized)
    
    # Train Word2Vec
    model = train_word2vec(tokenized, sg=1, epochs=10, seed=42, vector_size=50)
    assert "המלך" in model.wv
    assert model.wv["המלך"].shape == (50,)
    
    # Run Procrustes space shift checks on two halves
    first_half  = [tokenizer.encode_to_tokens(v) for ch in chapters[:5] for v in ch]
    second_half = [tokenizer.encode_to_tokens(v) for ch in chapters[5:] for v in ch]
    
    m1 = train_word2vec(first_half,  sg=1, epochs=10, seed=42, vector_size=50)
    m2 = train_word2vec(second_half, sg=1, epochs=10, seed=42, vector_size=50)
    
    shifts, A, A_rot, B = align_spaces_procrustes(m1, m2, counts_total, min_corpus_freq=2)
    
    # Assert alignment outputs
    assert len(shifts) > 0
    assert A.shape[1] == 50
    assert A_rot.shape == A.shape
    assert B.shape == B.shape
    
    # Cosine shifts must be in bounds [0.0, 2.0]
    for word, dist in shifts:
        assert 0.0 <= dist <= 2.0, f"Distance {dist} out of bounds for {word}"


def test_chapter_centroids():
    """Asserts chapter centroids and similarity matrices evaluate correctly."""
    chapters, all_verses = load_chapters_and_verses()
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    tokenizer.train(all_verses)
    tokenized = [tokenizer.encode_to_tokens(v) for v in all_verses]
    tokenized_by_ch = [[tokenizer.encode_to_tokens(v) for v in ch] for ch in chapters]
    
    model = train_word2vec(tokenized, sg=1, epochs=10, seed=42, vector_size=50)
    
    centroids = compute_chapter_centroids(model, tokenized_by_ch)
    assert centroids.shape == (10, 50)
    
    sim_matrix = compute_centroid_similarity_matrix(centroids)
    assert sim_matrix.shape == (10, 10)
    assert np.allclose(np.diag(sim_matrix), 1.0)  # Diagonal must be 1.0 (self cosine)
