"""Unit tests for PyTorch ZereshMLP classifier."""

import numpy as np
import torch
from esther_nlp.data import load_chapters_and_verses
from esther_nlp.tokenizer import EstherBpeTokenizer
from esther_nlp.embeddings import train_word2vec
from esther_nlp.classifier import (
    ZereshMLP,
    build_classifier_examples,
    featurize_sentence,
    run_cross_validation
)


def test_classifier_feauturization_and_mlp():
    """Asserts classification datasets build with correct pooling sizes and MLPs train."""
    _, all_verses = load_chapters_and_verses()
    
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    tokenizer.train(all_verses)
    tokenized = [tokenizer.encode_to_tokens(v) for v in all_verses]
    
    model_w2v = train_word2vec(tokenized, sg=1, epochs=10, seed=42, vector_size=50)
    
    # Build examples
    examples = build_classifier_examples(all_verses, tokenizer)
    assert len(examples) > 0
    
    # Assert tuple structure
    masked, label, raw_v, glob_idx = examples[0]
    assert "[MASK]" in masked
    assert label in [0, 1]
    assert isinstance(raw_v, str)
    assert isinstance(glob_idx, int)
    
    # Featurize
    x_feat = featurize_sentence(masked, model_w2v, dim=50)
    # 3 * vector_size (mean, max, min) + 1 length = 151 dimensions
    assert x_feat.shape == (151,)
    assert isinstance(x_feat, np.ndarray)
    
    # Run MLP forward pass
    mlp = ZereshMLP(d_in=151, d_hid=4, n_cls=2)
    x_tensor = torch.tensor(np.expand_dims(x_feat, axis=0), dtype=torch.float32)
    
    mlp.eval()
    with torch.no_grad():
        logits = mlp(x_tensor)
    assert logits.shape == (1, 2)


def test_cross_validation_loop():
    """Asserts CV loop executes successfully without leakage crashes."""
    X = np.random.randn(20, 151).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)
    
    # Execute K-fold with tiny epochs for quick tests
    mean_acc, fold_accs, preds, confs = run_cross_validation(
        X, y, n_splits=3, random_state=42, epochs=5
    )
    
    assert len(fold_accs) == 3
    assert len(preds) == 20
    assert len(confs) == 20
    assert 0.0 <= mean_acc <= 1.0
