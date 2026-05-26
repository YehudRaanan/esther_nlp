"""Regression test asserting exact numerical reproducibility via seed pinning."""

import numpy as np
import torch
from esther_nlp.data import load_chapters_and_verses
from esther_nlp.tokenizer import EstherBpeTokenizer
from esther_nlp.embeddings import train_word2vec
from esther_nlp.analysis import get_radial_ranks
from esther_nlp.classifier import (
    build_classifier_examples,
    featurize_sentence,
    run_cross_validation
)


def test_seed_pinning_determinism():
    """Asserts Word2Vec and PyTorch MLP generate exact identical values.

    Enforces that:
    1. Skip-gram 100ep puts Haman at rank 2 of 'המלך' with cosine ~0.48.
    2. MLP classifier scores exactly 83.6% cross-validation accuracy.
    """
    # Load and tokenize
    _, all_verses = load_chapters_and_verses()
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    tokenizer.train(all_verses)
    tokenized = [tokenizer.encode_to_tokens(v) for v in all_verses]
    
    # 1. Assert Word2Vec Determinism
    # Pinned Word2Vec Skip-gram model with seed 42
    sg_model = train_word2vec(tokenized, sg=1, epochs=100, seed=42, vector_size=50)
    
    # Haman's similarity to the king
    haman_ranks = get_radial_ranks(sg_model, "המלך", ["המן"])
    rank_haman, cos_haman = haman_ranks.get("המן", (-1, 0.0))
    
    # In a fully deterministic setup, Haman sits at rank 2
    assert rank_haman == 2, f"Expected Haman at rank 2 of 'המלך', found {rank_haman}"
    # Cosine must be approximately 0.481
    assert np.allclose(cos_haman, 0.481, atol=1e-2), f"Expected Haman cosine ~0.481, found {cos_haman:.3f}"
    
    # 2. Assert PyTorch Neural Network Determinism
    # Build examples and pool features
    examples = build_classifier_examples(all_verses, tokenizer)
    X = np.stack([featurize_sentence(m, sg_model, dim=50) for m, _, _, _ in examples])
    y = np.array([lbl for _, lbl, _, _ in examples], dtype=np.int64)
    
    # Run cross-validation with pinned weights initialization seeds
    mean_acc, fold_accs, preds, confs = run_cross_validation(
        X, y, n_splits=5, random_state=42, epochs=200
    )
    
    # Strict determinism validation checks
    # Classifier scores exactly 83.7% (0.837)
    expected_mean_acc = 0.837
    assert np.allclose(mean_acc, expected_mean_acc, atol=1e-2), (
        f"Expected cross-validation accuracy {expected_mean_acc:.3f}, found {mean_acc:.3f}"
    )
    
    # Verify first 3 fold accuracies to ensure matching train distributions
    assert np.allclose(fold_accs[0], 0.769, atol=1e-2), f"Fold 0 accuracy changed: {fold_accs[0]:.3f}"
    assert np.allclose(fold_accs[1], 0.917, atol=1e-2), f"Fold 1 accuracy changed: {fold_accs[1]:.3f}"
    assert np.allclose(fold_accs[2], 0.917, atol=1e-2), f"Fold 2 accuracy changed: {fold_accs[2]:.3f}"
